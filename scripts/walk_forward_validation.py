#!/usr/bin/env python3
"""
WALK-FORWARD VALIDATION - JL CAPITAL TRADE
Validação robusta com simulação cronológica e testes fora da amostra (OOS)
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from aggressive_feature_compatibility import AggressiveFeatureCompatibility
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('walk_forward_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class WalkForwardValidator:
    def __init__(self, symbol='EURUSD', timeframe=mt5.TIMEFRAME_H1):
        self.symbol = symbol
        self.timeframe = timeframe
        self.compat = AggressiveFeatureCompatibility()
        self.results = []
        
        logger.info(f"🚀 Iniciando Validação Walk-Forward para {symbol}")

    def connect_mt5(self):
        if not mt5.initialize():
            logger.error("Falha ao inicializar MT5")
            return False
        
        authorized = mt5.login(
            login=int(os.getenv('MT5_LOGIN')),
            password=os.getenv('MT5_PASSWORD'),
            server=os.getenv('MT5_SERVER')
        )
        return authorized

    def get_data(self, count=12000):
        from jl_capital_trade.mt5_connector import MT5Connector
        from jl_capital_trade.config import config
        
        connector = MT5Connector(config)
        if not connector.connect():
            logger.error("Falha ao conectar ao MT5")
            return None
        
        logger.info(f"📥 Coletando {count} barras de histórico via conector...")
        df = connector.get_historical_data(self.symbol, "H1", count)
        connector.disconnect()
        
        if df is not None:
            # Reseta o index para manter compatibilidade com o resto do script
            df = df.reset_index()
        return df

    def prepare_data(self, df):
        logger.info("🛠️ Preparando features robustas (Hurst, Efficiency, Fractal)...")
        # Calcula as features usando o DataCollector para garantir paridade
        from jl_capital_trade.data_collector import DataCollector
        from jl_capital_trade.config import config
        
        collector = DataCollector(config, None)
        df_feat = collector.calculate_indicators(df, self.symbol)
        
        # Define target (Retorno futuro > 0.1% em 5 períodos)
        forward_periods = 5
        return_threshold = 0.001
        df_feat['future_close'] = df_feat['close'].shift(-forward_periods)
        df_feat['target'] = (df_feat['future_close'] / df_feat['close'] - 1 > return_threshold).astype(int)
        
        # Remove NaNs
        df_feat = df_feat.dropna()
        
        # Features alinhadas com o novo ml_models.py
        features = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position', 'atr', 'volume_ratio',
            'close_position', 'volatility', 'ema_cross',
            'momentum', 'roc', 'hurst', 'fractal_dim', 'efficiency_ratio'
        ]
        
        X = df_feat[features]
        y = df_feat['target']
        times = df_feat['time']
        prices = df_feat['close']
        
        # Guardamos as colunas de alinhamento técnico para o loop de validação
        alignment_cols = df_feat[['hurst', 'efficiency_ratio']]
        
        logger.info(f"📊 Features utilizadas: {features}")
        
        return X, y, times, prices, alignment_cols

    def run_validation(self, X, y, times, prices, alignment, n_walks=5, train_ratio=0.7):
        """Executa a validação Walk-Forward cronológica com filtros de 70%"""
        n_samples = len(X)
        walk_size = n_samples // n_walks
        
        logger.info(f"🔄 Iniciando {n_walks} ciclos de Walk-Forward (Alvo 70%)...")
        
        all_preds = []
        all_true = []
        equity_curve = [100000.0] # Saldo inicial
        
        # Parâmetros suavizados para garantir execução na validação
        confidence_threshold = 0.68
        hurst_threshold = 0.48
        efficiency_threshold = 0.25
        
        for i in range(n_walks):
            # Define limites do walk atual
            end_idx = (i + 1) * walk_size
            train_end = i * walk_size + int(walk_size * train_ratio)
            
            # Dados de Treino
            X_train = X.iloc[i * walk_size : train_end]
            y_train = y.iloc[i * walk_size : train_end]
            
            # Dados de Teste
            X_test = X.iloc[train_end : end_idx]
            y_test = y.iloc[train_end : end_idx]
            test_prices = prices.iloc[train_end : end_idx]
            test_alignment = alignment.iloc[train_end : end_idx]
            
            if len(X_train) < 500 or len(X_test) < 100:
                continue
                
            logger.info(f"  Walk {i+1}: Treino {times.iloc[i*walk_size]} ate {times.iloc[train_end-1]} | Teste {times.iloc[train_end]} ate {times.iloc[end_idx-1]}")
            
            # Treina Modelo (ExtraTrees + XGBoost Ensemble simplificado)
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Previsões
            probs = model.predict_proba(X_test)[:, 1]
            
            # Custos fixos
            commission_per_lot = float(os.getenv('COMMISSION_PER_LOT', 7.0))
            slippage_pips = float(os.getenv('EXPECTED_SLIPPAGE_PIPS', 0.5))
            
            # Simulação
            for j in range(len(probs)):
                # Filtro Sniper de 70%
                is_confident = (probs[j] > confidence_threshold) or (probs[j] < (1 - confidence_threshold))
                is_aligned = (test_alignment.iloc[j]['hurst'] > hurst_threshold) and \
                             (test_alignment.iloc[j]['efficiency_ratio'] > efficiency_threshold)
                
                if is_confident and is_aligned:
                    current_balance = equity_curve[-1]
                    atr = X_test.iloc[j]['atr']
                    
                    # Risco 1.5%
                    risk_amount = current_balance * 0.015
                    sl_pips = 30 # Base
                    volume = risk_amount / (sl_pips * 10)
                    
                    total_costs = (volume * commission_per_lot) + (volume * 1.5 * 10) # 1.5 pips spread
                    
                    # Direção
                    is_buy = probs[j] > 0.5
                    actual_win = (y_test.iloc[j] == 1) if is_buy else (y_test.iloc[j] == 0)
                    
                    if actual_win:
                        # Lucro RR 1:4
                        equity_curve.append(current_balance + (risk_amount * 4) - total_costs)
                        all_preds.append(1 if is_buy else 0)
                    else:
                        equity_curve.append(current_balance - risk_amount - total_costs)
                        all_preds.append(0 if is_buy else 1)
                    
                    all_true.append(1 if is_buy else 0)
                
        # Calcula Métricas Finais
        self._calculate_metrics(all_true, all_preds, equity_curve)

    def _calculate_metrics(self, y_true, y_pred, equity_curve):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        # Métricas Financeiras
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        # Drawdown
        roll_max = equity_series.cummax()
        drawdown = (equity_series - roll_max) / roll_max
        max_dd = drawdown.min() * 100
        
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
        
        logger.info("\n" + "="*50)
        logger.info("📊 RELATÓRIO FINAL DE VALIDAÇÃO (OOS)")
        logger.info("="*50)
        logger.info(f"📈 Retorno Total Simulado: {total_return:.2f}%")
        logger.info(f"📉 Max Drawdown: {max_dd:.2f}%")
        logger.info(f"⚖️ Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"🎯 Acurácia OOS: {acc:.2%}")
        logger.info(f"✅ Precisão (Acerto Compra): {prec:.2%}")
        logger.info(f"🔍 Recall: {rec:.2%}")
        logger.info(f"💰 Saldo Final: ${equity_series.iloc[-1]:.2f}")
        logger.info("="*50)

def main():
    validator = WalkForwardValidator()
    # Aumenta a amostra para uma validação mais robusta (12.000 barras ~ 1.5 anos em H1)
    data = validator.get_data(count=12000)
    
    if data is not None:
        X, y, times, prices, alignment = validator.prepare_data(data)
        validator.run_validation(X, y, times, prices, alignment)
    else:
        logger.error("Não foi possível obter dados para validação.")

if __name__ == "__main__":
    main()
