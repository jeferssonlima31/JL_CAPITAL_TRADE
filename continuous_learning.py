#!/usr/bin/env python3
"""
SISTEMA DE APRENDIZADO CONTÍNUO - JL CAPITAL TRADE
Melhora continuamente a confiança dos modelos com dados em tempo real
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from integration_system import IntegrationSystem
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import joblib
import logging
import os
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class ContinuousLearningSystem:
    def __init__(self):
        self.data_buffer = []
        self.max_buffer_size = 1000
        self.retraining_interval = timedelta(hours=24)  # Retreina a cada 24h
        self.last_retraining = None
        self.models = {}
        self.ensemble_model = None
        
        # Configurações
        self.mt5_login = int(os.getenv('MT5_LOGIN', 3263303))
        self.mt5_password = os.getenv('MT5_PASSWORD', '!rH5UiSb')
        self.mt5_server = os.getenv('MT5_SERVER', 'Just2Trade-MT5')
        
        # Símbolos para coleta
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        self.timeframes = [mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]
        
        logger.info("🚀 Sistema de Aprendizado Contínuo Inicializado")
    
    def connect_to_mt5(self):
        """Conecta ao MT5"""
        try:
            if not mt5.initialize():
                logger.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
                return False
            
            authorized = mt5.login(
                login=self.mt5_login,
                password=self.mt5_password,
                server=self.mt5_server
            )
            
            if authorized:
                logger.info("✅ Conectado ao MT5 para coleta de dados")
                return True
            else:
                logger.error(f"Falha no login: {mt5.last_error()}")
                return False
                
        except Exception as e:
            logger.error(f"Erro na conexão: {e}")
            return False
    
    def collect_real_time_data(self):
        """Coleta dados em tempo real para aprendizado"""
        if not self.connect_to_mt5():
            return
        
        try:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    self._collect_symbol_data(symbol, timeframe)
                    time.sleep(0.1)  # Pequena pausa
                    
        finally:
            mt5.shutdown()
    
    def _collect_symbol_data(self, symbol, timeframe):
        """Coleta dados de um símbolo específico"""
        try:
            # Coleta dados históricos
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 500)
            if rates is None:
                return
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Calcula target (retorno futuro)
            df = self._calculate_features_and_target(df)
            
            # Adiciona ao buffer
            self._add_to_buffer(df)
            
            logger.info(f"📊 Dados coletados: {symbol} {timeframe} - {len(df)} registros")
            
        except Exception as e:
            logger.error(f"Erro coletando {symbol}: {e}")
    
    def _calculate_features_and_target(self, df):
        """Calcula features e target para aprendizado"""
        if len(df) < 100:
            return df
        
        # Features (mesmas do sistema principal)
        df = self._calculate_technical_indicators(df)
        
        # Target: Retorno 5 períodos à frente
        future_periods = 5
        df['future_close'] = df['close'].shift(-future_periods)
        df = df.dropna(subset=['future_close'])
        
        # Target binário: Retorno > 0.05%
        return_threshold = 0.0005
        df['target'] = (df['future_close'] / df['close'] - 1 > return_threshold).astype(int)
        
        return df
    
    def _calculate_technical_indicators(self, df):
        """Calcula indicadores técnicos"""
        # Médias móveis
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['ma20']
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Outros indicadores
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
        df['momentum'] = df['close'] - df['close'].shift(4)
        df['roc'] = (df['close'] / df['close'].shift(10) - 1) * 100
        df['price_vs_ma'] = (df['close'] / df['ma20'] - 1) * 100
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def _add_to_buffer(self, new_data):
        """Adiciona dados ao buffer de treinamento"""
        if len(self.data_buffer) >= self.max_buffer_size:
            # Remove os mais antigos
            self.data_buffer = self.data_buffer[len(new_data):]
        
        self.data_buffer.extend(new_data.to_dict('records'))
        
        logger.info(f"📦 Buffer: {len(self.data_buffer)}/{self.max_buffer_size} registros")
    
    def get_training_data(self):
        """Prepara dados para treinamento"""
        if len(self.data_buffer) < 100:
            return None, None
        
        df = pd.DataFrame(self.data_buffer)
        df = df.dropna()
        
        if len(df) < 50:
            return None, None
        
        # Features para treinamento
        feature_columns = [
            'open', 'high', 'low', 'close', 'tick_volume', 'spread',
            'returns', 'log_returns', 'ma5', 'ma20', 'ma50', 'ma200',
            'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width',
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'volatility',
            'atr', 'momentum', 'roc', 'price_vs_ma', 'high_low_ratio'
        ]
        
        # Filtra colunas disponíveis
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features]
        y = df['target']
        
        return X, y
    
    def train_individual_models(self, X, y):
        """Treina modelos individuais"""
        logger.info("🎯 Treinando modelos individuais...")
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Random Forest (substituto para LightGBM)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        # Logistic Regression
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        # Treina todos os modelos
        models = {
            'xgb': xgb_model,
            'rf': rf_model,
            'lr': lr_model
        }
        
        for name, model in models.items():
            logger.info(f"   🔧 Treinando {name}...")
            model.fit(X, y)
            
            # Validação cruzada
            cv_scores = self.cross_validate_model(model, X, y)
            logger.info(f"      ✅ {name} - Acurácia: {cv_scores['accuracy']:.3f}")
            
            self.models[name] = model
        
        return models
    
    def cross_validate_model(self, model, X, y):
        """Validação cruzada para time series"""
        tscv = TimeSeriesSplit(n_splits=5)
        
        accuracies, precisions, recalls, f1s = [], [], [], []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
        
        return {
            'accuracy': np.mean(accuracies),
            'precision': np.mean(precisions), 
            'recall': np.mean(recalls),
            'f1': np.mean(f1s)
        }
    
    def create_ensemble_model(self):
        """Cria modelo ensemble dos melhores modelos"""
        if not self.models:
            return None
        
        logger.info("🤖 Criando modelo ensemble...")
        
        # Usa os melhores modelos para o ensemble
        estimators = []
        for name, model in self.models.items():
            estimators.append((name, model))
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Soft voting para probabilidades
            n_jobs=-1
        )
        
        # Treina o ensemble
        X, y = self.get_training_data()
        if X is not None and y is not None:
            ensemble.fit(X, y)
            
            # Avalia ensemble
            cv_scores = self.cross_validate_model(ensemble, X, y)
            logger.info(f"   🎯 Ensemble - Acurácia: {cv_scores['accuracy']:.3f}")
            logger.info(f"   📊 Ensemble - F1-Score: {cv_scores['f1']:.3f}")
            
            self.ensemble_model = ensemble
            return ensemble
        
        return None
    
    def save_models(self):
        """Salva os modelos treinados"""
        if not os.path.exists('trained_models_continuous'):
            os.makedirs('trained_models_continuous')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salva modelos individuais
        for name, model in self.models.items():
            filename = f'trained_models_continuous/{name}_model_{timestamp}.joblib'
            joblib.dump(model, filename)
            logger.info(f"💾 Modelo {name} salvo: {filename}")
        
        # Salva ensemble
        if self.ensemble_model:
            ensemble_filename = f'trained_models_continuous/ensemble_model_{timestamp}.joblib'
            joblib.dump(self.ensemble_model, ensemble_filename)
            logger.info(f"💾 Ensemble salvo: {ensemble_filename}")
            
            # Também salva como modelo principal
            main_filename = 'trained_models/xgboost_continuous_learning.model'
            if hasattr(self.ensemble_model, 'predict_proba'):
                # Se for ensemble, salva o melhor modelo individual
                best_model = self.models.get('xgb')
                if best_model:
                    best_model.save_model(main_filename)
                    logger.info(f"⭐ Melhor modelo salvo como principal: {main_filename}")
    
    def should_retrain(self):
        """Verifica se deve retreinar"""
        if self.last_retraining is None:
            return True
        
        return datetime.now() - self.last_retraining >= self.retraining_interval
    
    def run_retraining(self):
        """Executa ciclo completo de retreinamento"""
        if not self.should_retrain():
            logger.info("⏭️ Ainda não é hora de retreinar")
            return
        
        logger.info("\n" + "="*60)
        logger.info("🚀 INICIANDO RETREINAMENTO DOS MODELOS")
        logger.info("="*60)
        
        # Coleta dados atualizados
        self.collect_real_time_data()
        
        # Prepara dados de treinamento
        X, y = self.get_training_data()
        if X is None or y is None:
            logger.warning("⚠️ Dados insuficientes para treinamento")
            return
        
        logger.info(f"📊 Dados para treinamento: {X.shape[0]} amostras, {X.shape[1]} features")
        logger.info(f"🎯 Distribuição do target: {y.value_counts().to_dict()}")
        
        # Treina modelos
        self.train_individual_models(X, y)
        
        # Cria ensemble
        self.create_ensemble_model()
        
        # Salva modelos
        self.save_models()
        
        self.last_retraining = datetime.now()
        logger.info(f"✅ Retreinamento concluído em {self.last_retraining}")
    
    def monitor_model_performance(self):
        """Monitora performance dos modelos em tempo real"""
        # Implementar monitoramento contínuo
        pass

def main():
    """Função principal do aprendizado contínuo"""
    learning_system = ContinuousLearningSystem()
    
    try:
        # Executa retreinamento inicial
        learning_system.run_retraining()
        
        logger.info("\n🔍 Sistema de aprendizado contínuo pronto!")
        logger.info("💡 Executar periodicamente para melhorar os modelos")
        
    except Exception as e:
        logger.error(f"❌ Erro no sistema de aprendizado: {e}")

if __name__ == "__main__":
    main()