#!/usr/bin/env python3
"""
OTIMIZAÇÃO TURBO - JL CAPITAL TRADE
Versão acelerada com busca inteligente de parâmetros
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from integration_system import IntegrationSystem
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import logging
import os
from datetime import datetime
import time
from dotenv import load_dotenv

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('turbo_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class TurboOptimizer:
    def __init__(self):
        self.system = IntegrationSystem()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_confidence = 0.0
        
        logger.info("⚡ Otimizador Turbo Inicializado")
    
    def connect_to_mt5(self):
        """Conecta ao MT5"""
        try:
            if not mt5.initialize():
                return False
            
            authorized = mt5.login(
                login=int(os.getenv('MT5_LOGIN', 3263303)),
                password=os.getenv('MT5_PASSWORD', '!rH5UiSb'),
                server=os.getenv('MT5_SERVER', 'Just2Trade-MT5')
            )
            
            return authorized
                
        except Exception as e:
            logger.error(f"Erro na conexão: {e}")
            return False
    
    def collect_data_turbo(self, symbol='EURUSD', bars=500):
        """Coleta dados de forma rápida"""
        if not self.connect_to_mt5():
            return None
        
        try:
            # Coleta apenas do timeframe H1 (mais rápido)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
            if rates is None:
                return None
                
            df = pd.DataFrame(rates)
            df['symbol'] = symbol
            
            # Calcula features essenciais
            df = self._calculate_turbo_features(df)
            
            # Define target com threshold mais realista
            df = self._define_turbo_target(df, return_threshold=0.002)
            
            logger.info(f"📊 {symbol} H1: {len(df)} bars coletados")
            
            return df.dropna()
            
        finally:
            mt5.shutdown()
    
    def _calculate_turbo_features(self, df):
        """Calcula apenas features mais importantes"""
        # Features básicas
        df['returns'] = df['close'].pct_change()
        
        # Médias móveis essenciais
        for window in [5, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD simples
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volatilidade
        df['volatility_20'] = df['close'].rolling(20).std()
        
        return df
    
    def _define_turbo_target(self, df, return_threshold=0.005, forward_periods=3):
        """Define target mais balanceado"""
        # Retorno futuro mais curto
        df['future_close'] = df['close'].shift(-forward_periods)
        df = df.dropna(subset=['future_close'])
        
        # Target binário com threshold mais realista
        df['target'] = (df['future_close'] / df['close'] - 1 > return_threshold).astype(int)
        
        logger.info(f"🎯 Distribuição target: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def optimize_turbo(self, X, y):
        """Otimização turbo com busca aleatória"""
        logger.info("⚡ Otimização Turbo (Randomized Search)")
        
        # Parâmetros para busca aleatória (mais rápido)
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # TimeSeries Cross Validation
        tscv = TimeSeriesSplit(n_splits=3)  # Menos splits para mais velocidade
        
        # Modelo base
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
        
        # Randomized Search (mais rápido que Grid Search)
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=20,  # Apenas 20 combinações em vez de 2916
            scoring='accuracy',
            cv=tscv,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        logger.info("🔍 Executando Randomized Search (muito mais rápido)...")
        random_search.fit(X, y)
        
        logger.info(f"✅ Melhores parâmetros: {random_search.best_params_}")
        logger.info(f"✅ Melhor score: {random_search.best_score_:.3f}")
        
        # Modelo com calibração
        best_model = random_search.best_estimator_
        calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=3)
        calibrated_model.fit(X, y)
        
        return calibrated_model
    
    def train_turbo_ensemble(self, X, y):
        """Treina ensemble turbo"""
        logger.info("🤖 Treinando Ensemble Turbo")
        
        # Modelos rápidos
        models = {
            'xgb_turbo': self.optimize_turbo(X, y),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gbc': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Treina modelos rápidos
        for name, model in models.items():
            if name != 'xgb_turbo':  # xgb já foi treinado
                logger.info(f"🔧 Treinando {name}...")
                model.fit(X, y)
        
        # Avalia confiança
        confidence_scores = {}
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X)
            confidence = np.mean(np.max(y_pred_proba, axis=1))
            confidence_scores[name] = confidence
            logger.info(f"   📊 {name} - Confiança média: {confidence:.3f}")
        
        # Seleciona melhor modelo
        best_model_name = max(confidence_scores, key=confidence_scores.get)
        best_model = models[best_model_name]
        self.best_confidence = confidence_scores[best_model_name]
        
        logger.info(f"⭐ Melhor modelo: {best_model_name} - Confiança: {self.best_confidence:.3f}")
        
        return best_model
    
    def run_turbo_optimization(self):
        """Executa otimização turbo"""
        logger.info("\n" + "="*60)
        logger.info("⚡ INICIANDO OTIMIZAÇÃO TURBO")
        logger.info("="*60)
        
        # Coleta dados rápidos
        logger.info("📦 Coletando dados turbo...")
        data = self.collect_data_turbo()
        
        if data is None or len(data) < 100:
            logger.error("❌ Dados insuficientes")
            return False
        
        # Prepara features
        feature_columns = [col for col in data.columns if col not in 
                          ['time', 'target', 'future_close', 'symbol']]
        X = data[feature_columns].select_dtypes(include=[np.number])
        y = data['target']
        
        logger.info(f"📊 Dados: {X.shape[0]} amostras, {X.shape[1]} features")
        logger.info(f"🎯 Target balance: {y.value_counts().to_dict()}")
        
        # Normaliza
        X_scaled = self.scaler.fit_transform(X)
        
        # Otimização turbo
        optimized_model = self.train_turbo_ensemble(X_scaled, y)
        
        # Salva modelo
        self._save_turbo_model(optimized_model, X.columns.tolist())
        
        # Testa confiança
        self.test_turbo_confidence(optimized_model, X_scaled, y)
        
        return True
    
    def _save_turbo_model(self, model, feature_names):
        """Salva modelo turbo"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salva modelo
        model_path = f'trained_models/xgboost_turbo_{timestamp}.model'
        if hasattr(model, 'save_model'):
            model.save_model(model_path)
        else:
            joblib.dump(model, model_path.replace('.model', '.joblib'))
        
        # Salva scaler
        scaler_path = f'trained_models/scaler_turbo_{timestamp}.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"💾 Modelo turbo salvo: {model_path}")
        logger.info(f"💾 Scaler turbo salvo: {scaler_path}")
        logger.info(f"🎯 Confiança turbo: {self.best_confidence:.3f}")
    
    def test_turbo_confidence(self, model, X_test, y_test):
        """Testa confiança turbo"""
        y_pred_proba = model.predict_proba(X_test)
        confidences = np.max(y_pred_proba, axis=1)
        
        mean_confidence = np.mean(confidences)
        high_confidence = np.sum(confidences >= 0.7) / len(confidences)
        
        logger.info("\n" + "="*40)
        logger.info("📊 TESTE TURBO DE CONFIANÇA")
        logger.info("="*40)
        logger.info(f"📈 Confiança média: {mean_confidence:.3f}")
        logger.info(f"🎯 Sinais de alta confiança (>70%): {high_confidence:.1%}")
        
        # Performance
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"✅ Acurácia: {accuracy:.3f}")
        
        if mean_confidence > 0.6:
            logger.info("🎉 CONFIANÇA TURBO OTIMIZADA!")
        else:
            logger.info("⚠️ Confiança ainda precisa melhorar")

def main():
    """Função principal"""
    optimizer = TurboOptimizer()
    
    try:
        success = optimizer.run_turbo_optimization()
        
        if success:
            logger.info("\n✅ OTIMIZAÇÃO TURBO CONCLUÍDA!")
            logger.info("⚡ Confiança melhorada em minutos em vez de horas")
        else:
            logger.error("❌ Falha na otimização turbo")
            
    except Exception as e:
        logger.error(f"❌ Erro na otimização turbo: {e}")

if __name__ == "__main__":
    main()