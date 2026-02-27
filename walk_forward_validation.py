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

    def get_data(self, count=10000):
        if not self.connect_mt5():
            return None
        
        logger.info(f"📥 Coletando {count} barras de histórico...")
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
        mt5.shutdown()
        
        if rates is None:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def prepare_data(self, df):
        logger.info("🛠️ Preparando features robustas e targets...")
        # Calcula as features
        df_feat = df.copy()
        self.compat._calculate_all_features(df_feat)
        
        # Define target (Retorno futuro > 0.1% em 5 períodos)
        forward_periods = 5
        return_threshold = 0.001
        df_feat['future_close'] = df_feat['close'].shift(-forward_periods)
        df_feat['target'] = (df_feat['future_close'] / df_feat['close'] - 1 > return_threshold).astype(int)
        
        # Remove NaNs
        df_feat = df_feat.dropna()
        
        # Seleciona apenas as 6 features robustas
        features = self.compat.required_features
        X = df_feat[features]
        y = df_feat['target']
        times = df_feat['time']
        prices = df_feat['close']
        
        logger.info(f"📊 Features utilizadas: {features}")
        
        return X, y, times, prices

    def run_validation(self, X, y, times, prices, n_walks=5, train_ratio=0.7):
        """Executa a validação Walk-Forward cronológica"""
        n_samples = len(X)
        walk_size = n_samples // n_walks
        
        logger.info(f"🔄 Iniciando {n_walks} ciclos de Walk-Forward...")
        
        all_preds = []
        all_true = []
        equity_curve = [100000.0] # Saldo inicial
        
        for i in range(n_walks):
            # Define limites do walk atual
            end_idx = (i + 1) * walk_size
            train_end = int(end_idx * train_ratio)
            
            # Dados de Treino
            X_train = X.iloc[i * walk_size : train_end]
            y_train = y.iloc[i * walk_size : train_end]
            
            # Dados de Teste (Fora da amostra para este walk)
            X_test = X.iloc[train_end : end_idx]
            y_test = y.iloc[train_end : end_idx]
            test_prices = prices.iloc[train_end : end_idx]
            
            if len(X_train) < 100 or len(X_test) < 20:
                continue
                
            logger.info(f"  Walk {i+1}: Treino {times.iloc[i*walk_size]} ate {times.iloc[train_end-1]} | Teste {times.iloc[train_end]} ate {times.iloc[end_idx-1]}")
            
            # Treina Modelo (ExtraTrees como no sistema agressivo)
            model = ExtraTreesClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Previsões
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
            # Custos fixos da simulação
            commission_per_lot = float(os.getenv('COMMISSION_PER_LOT', 7.0))
            slippage_pips = float(os.getenv('EXPECTED_SLIPPAGE_PIPS', 0.5))
            avg_spread = 1.0 # Spread médio para simulação
            
            # Simulação Cronológica de Lucro/Perda com CUSTOS REAIS
            for j in range(len(preds)):
                if probs[j] > 0.7: # Filtro de confiança agressivo
                    current_balance = equity_curve[-1]
                    
                    # Calcula volume baseado no risco de 1.5%
                    risk_amount = current_balance * 0.015
                    sl_pips = 30
                    volume = risk_amount / (sl_pips * 10) # $10 por pip
                    
                    # Custos Totais (Comissão + Spread + Slippage)
                    total_costs = (volume * commission_per_lot) + (volume * (avg_spread + slippage_pips) * 10)
                    
                    if y_test.iloc[j] == 1:
                        # Lucro Bruto (RR 1:4 = 120 pips)
                        gross_profit = current_balance * 0.015 * 4
                        equity_curve.append(current_balance + gross_profit - total_costs)
                    else:
                        # Perda Bruta (30 pips)
                        gross_loss = current_balance * 0.015
                        equity_curve.append(current_balance - gross_loss - total_costs)
                
            all_preds.extend(preds)
            all_true.extend(y_test)
            
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
    data = validator.get_data(count=8000)
    
    if data is not None:
        X, y, times, prices = validator.prepare_data(data)
        validator.run_validation(X, y, times, prices)
    else:
        logger.error("Não foi possível obter dados para validação.")

if __name__ == "__main__":
    main()
