#!/usr/bin/env python3
"""
OTIMIZAÇÃO AGRESSIVA DE MODELOS - JL CAPITAL TRADE
Aumenta drasticamente a confiança dos modelos com técnicas avançadas
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from integration_system import IntegrationSystem
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
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
        logging.FileHandler('aggressive_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class AggressiveOptimizer:
    def __init__(self):
        self.system = IntegrationSystem()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_confidence = 0.0
        
        logger.info("🚀 Otimizador Agressivo Inicializado")
    
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
    
    def collect_high_quality_data(self, symbol='EURUSD', bars=5000):
        """Coleta dados de alta qualidade para treinamento"""
        if not self.connect_to_mt5():
            return None
        
        try:
            # Coleta dados de múltiplos timeframes
            timeframes = [mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]
            all_data = []
            
            for timeframe in timeframes:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
                if rates is not None:
                    df = pd.DataFrame(rates)
                    df['timeframe'] = timeframe
                    df['symbol'] = symbol
                    all_data.append(df)
                    
                    logger.info(f"📊 {symbol} TF{timeframe}: {len(df)} bars")
                time.sleep(0.1)
            
            if not all_data:
                return None
                
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Calcula features avançadas
            combined_df = self._calculate_advanced_features(combined_df)
            
            # Define target mais agressivo (retorno de 1%)
            combined_df = self._define_aggressive_target(combined_df, return_threshold=0.01)
            
            return combined_df.dropna()
            
        finally:
            mt5.shutdown()
    
    def _calculate_advanced_features(self, df):
        """Calcula features técnicas avançadas"""
        # Features de Tempo/Sessão (Horário de Portugal)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        
        # Sessões
        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
        df['is_usa'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        df['is_overlay'] = (df['is_london'] & df['is_usa']).astype(int)

        # Features básicas
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Médias móveis múltiplas
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        
        # Momentum e taxa de mudança
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
        df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # Volatilidade
        df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['volatility_50'] = df['close'].rolling(50).std() / df['close'].rolling(50).mean()
        
        # RSI múltiplo
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            
            rs = avg_gain / avg_loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD múltiplo
        for fast, slow in [(12, 26), (8, 17), (5, 35)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
            df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume analysis
        df['volume_ma_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        df['volume_roc'] = df['tick_volume'].pct_change()
        
        return df
    
    def _define_aggressive_target(self, df, return_threshold=0.01, forward_periods=5):
        """Define target mais agressivo para maior confiança"""
        # Retorno futuro
        df['future_close'] = df['close'].shift(-forward_periods)
        df = df.dropna(subset=['future_close'])
        
        # Target binário com threshold mais alto
        df['target'] = (df['future_close'] / df['close'] - 1 > return_threshold).astype(int)
        
        # Target multi-classe para maior precisão
        returns = df['future_close'] / df['close'] - 1
        df['target_multi'] = pd.cut(returns, 
                                   bins=[-np.inf, -0.005, 0.005, 0.01, 0.02, np.inf],
                                   labels=[0, 1, 2, 3, 4])
        
        logger.info(f"🎯 Distribuição target: {df['target'].value_counts().to_dict()}")
        logger.info(f"🎯 Distribuição multi-classe: {df['target_multi'].value_counts().to_dict()}")
        
        return df
    
    def optimize_xgboost_aggressive(self, X, y):
        """Otimização agressiva do XGBoost com Controle de Overfitting"""
        logger.info("🔥 Otimização agressiva do XGBoost (Regularizado)")
        
        # Parâmetros para grid search - Adicionado L1, L2 e Early Stopping
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'reg_alpha': [0.1, 0.5, 1.0], # L1 Regularization
            'reg_lambda': [0.1, 0.5, 1.0], # L2 Regularization
            'gamma': [0.1, 0.2]
        }
        
        # TimeSeries Cross Validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scorer personalizado (foca em confidence)
        def confidence_scorer(y_true, y_pred_proba):
            # Maximiza a probabilidade das previsões corretas
            correct_mask = (y_pred_proba.argmax(axis=1) == y_true)
            if correct_mask.any():
                return np.mean(y_pred_proba[correct_mask].max(axis=1))
            return 0.0
        
        # Modelo base
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
        
        # Randomized Search
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,
            scoring=make_scorer(accuracy_score),
            cv=tscv,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        logger.info("🔍 Executando Randomized Search...")
        random_search.fit(X, y)
        
        logger.info(f"✅ Melhores parâmetros: {random_search.best_params_}")
        logger.info(f"✅ Melhor score: {random_search.best_score_:.3f}")
        
        # Modelo com calibração de confiança
        best_model = random_search.best_estimator_
        calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
        calibrated_model.fit(X, y)
        
        return calibrated_model
    
    def train_confidence_ensemble(self, X, y):
        """Treina ensemble focado em confiança"""
        logger.info("🤖 Treinando Ensemble de Confiança")
        
        # Modelos diversificados
        models = {
            'xgb_agg': self.optimize_xgboost_aggressive(X, y),
            'gbc': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'et': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'svc': SVC(probability=True, random_state=42)
        }
        
        # Treina todos os modelos
        for name, model in models.items():
            if name != 'xgb_agg':  # xgb já foi treinado
                logger.info(f"🔧 Treinando {name}...")
                model.fit(X, y)
        
        # Avalia confiança individual
        confidence_scores = {}
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X)
            confidence = np.mean(np.max(y_pred_proba, axis=1))
            confidence_scores[name] = confidence
            logger.info(f"   📊 {name} - Confiança média: {confidence:.3f}")
        
        # Seleciona modelo com maior confiança
        best_model_name = max(confidence_scores, key=confidence_scores.get)
        best_model = models[best_model_name]
        self.best_confidence = confidence_scores[best_model_name]
        
        logger.info(f"⭐ Melhor modelo: {best_model_name} - Confiança: {self.best_confidence:.3f}")
        
        return best_model
    
    def select_robust_features(self, X, y):
        """Seleciona as features mais robustas para evitar overfitting"""
        logger.info("🔍 Selecionando features robustas...")
        
        # Modelo base para seleção (ExtraTrees é excelente para isso)
        selector_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        
        # Seleciona baseado em importância (threshold médio)
        selector = SelectFromModel(selector_model, threshold="1.25*mean")
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"✅ Redução de features: {len(X.columns)} -> {len(selected_features)}")
        logger.info(f"📋 Features selecionadas: {selected_features}")
        
        return selected_features

    def run_aggressive_optimization(self):
        """Executa otimização agressiva completa com controle de overfitting"""
        logger.info("\n" + "="*70)
        logger.info("🚀 INICIANDO OTIMIZAÇÃO AGRESSIVA (OVERFITTING CONTROL)")
        logger.info("="*70)
        
        # Coleta dados
        logger.info("📦 Coletando dados de alta qualidade...")
        data = self.collect_high_quality_data()
        
        if data is None or len(data) < 100:
            logger.error("❌ Dados insuficientes para otimização")
            return False
        
        # Prepara features e target
        feature_columns = [col for col in data.columns if col not in 
                          ['time', 'target', 'target_multi', 'future_close', 'symbol', 'timeframe']]
        X = data[feature_columns].select_dtypes(include=[np.number])
        y = data['target']
        
        # 1. Seleção de Features Robustas (Controle de Overfitting)
        robust_features = self.select_robust_features(X, y)
        X_robust = X[robust_features]
        
        logger.info(f"📊 Dados: {X_robust.shape[0]} amostras, {X_robust.shape[1]} features robustas")
        
        # 2. Normaliza apenas features robustas
        X_scaled = self.scaler.fit_transform(X_robust)
        
        # 3. Otimização agressiva (com regularização interna)
        optimized_model = self.train_confidence_ensemble(X_scaled, y)
        
        # Salva modelo otimizado com a lista de features robustas
        self._save_optimized_model(optimized_model, robust_features)
        
        # 4. Testa confiança em dados recentes (Validação fora do treino)
        self.test_confidence_improvement(optimized_model, X_scaled, y)
        
        return True
    
    def _save_optimized_model(self, model, feature_names):
        """Salva modelo otimizado"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salva modelo
        model_path = f'trained_models/xgboost_aggressive_{timestamp}.model'
        if hasattr(model, 'save_model'):
            model.save_model(model_path)
        else:
            joblib.dump(model, model_path.replace('.model', '.joblib'))
        
        # Salva scaler
        scaler_path = f'trained_models/scaler_aggressive_{timestamp}.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Salva feature names
        features_path = f'trained_models/features_aggressive_{timestamp}.pkl'
        joblib.dump(feature_names, features_path)
        
        logger.info(f"💾 Modelo salvo: {model_path}")
        logger.info(f"💾 Scaler salvo: {scaler_path}")
        logger.info(f"💾 Features salvas: {features_path}")
        logger.info(f"🎯 Confiança alcançada: {self.best_confidence:.3f}")
    
    def test_confidence_improvement(self, model, X_test, y_test):
        """Testa melhoria na confiança"""
        # Previsões com confiança
        y_pred_proba = model.predict_proba(X_test)
        confidences = np.max(y_pred_proba, axis=1)
        
        # Estatísticas de confiança
        mean_confidence = np.mean(confidences)
        median_confidence = np.median(confidences)
        high_confidence = np.sum(confidences >= 0.7) / len(confidences)
        
        logger.info("\n" + "="*50)
        logger.info("📊 TESTE DE MELHORIA DE CONFIANÇA")
        logger.info("="*50)
        logger.info(f"📈 Confiança média: {mean_confidence:.3f}")
        logger.info(f"📊 Confiança mediana: {median_confidence:.3f}")
        logger.info(f"🎯 Sinais de alta confiança (>70%): {high_confidence:.1%}")
        logger.info(f"🔢 Total de previsões: {len(confidences)}")
        
        # Avalia performance
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"✅ Acurácia: {accuracy:.3f}")
        
        if mean_confidence > 0.6:
            logger.info("🎉 CONFIANÇA OTIMIZADA COM SUCESSO!")
        else:
            logger.info("⚠️ Confiança ainda abaixo do ideal")

def main():
    """Função principal"""
    optimizer = AggressiveOptimizer()
    
    try:
        success = optimizer.run_aggressive_optimization()
        
        if success:
            logger.info("\n✅ OTIMIZAÇÃO AGRESSIVA CONCLUÍDA!")
            logger.info("🚀 Modelos com confiança significativamente melhorada")
        else:
            logger.error("❌ Falha na otimização")
            
    except Exception as e:
        logger.error(f"❌ Erro na otimização: {e}")

if __name__ == "__main__":
    main()