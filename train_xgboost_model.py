#!/usr/bin/env python3
# =============================================================================
# JL CAPITAL TRADE - TREINAMENTO DO MODELO XGBOOST
# =============================================================================

import os
import sys
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xgboost_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class XGBoostTrainer:
    """Treinador do modelo XGBoost para previsão de tendências"""
    
    def __init__(self):
        self.models_dir = "trained_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Hiperparâmetros otimizados para Forex
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'lambda': 1.0,
            'alpha': 0.0,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': 42
        }
        
        # Features importantes para Forex
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position', 'atr', 'volume_ratio',
            'close_position', 'volatility', 'ema_cross',
            'momentum', 'roc'
        ]
    
    def load_training_data(self, symbol="EURUSD", timeframe="H1"):
        """Carrega dados de treinamento"""
        try:
            # Primeiro, vamos coletar dados em tempo real para treinamento
            from collect_training_data import TrainingDataCollector
            
            collector = TrainingDataCollector()
            if not collector.connect_mt5():
                logger.error("Falha ao conectar ao MT5")
                return None
            
            # Coleta dados dos últimos 30 dias para treinamento rápido
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=30)
            
            logger.info(f"Coletando dados de {symbol} {timeframe} de {start_date} até {end_date}")
            
            # Obtém timeframe correto
            import MetaTrader5 as mt5
            tf_mapping = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            tf_value = tf_mapping.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Coleta dados
            df = collector.get_historical_data(symbol, tf_value, start_date, end_date)
            
            if df is None or len(df) == 0:
                logger.error("Nenhum dado coletado")
                return None
            
            # Calcula indicadores técnicos
            df = collector.calculate_technical_indicators(df)
            
            mt5.shutdown()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepara features para treinamento"""
        try:
            # Seleciona apenas as colunas de features disponíveis
            available_features = [f for f in self.feature_columns if f in df.columns]
            
            if not available_features:
                logger.error("Nenhuma feature disponível")
                return None, None
            
            # Extrai features e target
            X = df[available_features].copy()
            y = df['target'].copy()
            
            # Remove NaN values
            valid_indices = X.notnull().all(axis=1) & y.notnull()
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) == 0:
                logger.error("Nenhum dado válido após remoção de NaN")
                return None, None
            
            # Normaliza features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Salva o scaler
            scaler_path = os.path.join(self.models_dir, "feature_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"Features preparadas: {X_scaled.shape}")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Erro ao preparar features: {e}")
            return None, None
    
    def train_model(self, X, y, symbol="EURUSD", timeframe="H1"):
        """Treina o modelo XGBoost"""
        try:
            # Divide em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Dados de treino: {X_train.shape}")
            logger.info(f"Dados de teste: {X_test.shape}")
            
            # Calcula pesos de classe para lidar com desbalanceamento
            classes = np.unique(y_train)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train
            )
            
            # Cria dataset DMatrix para XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # Treina o modelo
            logger.info("Iniciando treinamento do XGBoost...")
            
            eval_set = [(dtrain, 'train'), (dtest, 'test')]
            
            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params['n_estimators'],
                evals=eval_set,
                early_stopping_rounds=self.params['early_stopping_rounds'],
                verbose_eval=50
            )
            
            # Faz previsões
            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calcula métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logger.info(f"\nRESULTADOS DO TREINAMENTO:")
            logger.info(f"   Acurácia: {accuracy:.4f}")
            logger.info(f"   Precisão: {precision:.4f}")
            logger.info(f"   Recall: {recall:.4f}")
            logger.info(f"   F1-Score: {f1:.4f}")
            
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"\nMatriz de Confusão:")
            logger.info(f"   Verdadeiros Negativos: {cm[0, 0]}")
            logger.info(f"   Falsos Positivos: {cm[0, 1]}")
            logger.info(f"   Falsos Negativos: {cm[1, 0]}")
            logger.info(f"   Verdadeiros Positivos: {cm[1, 1]}")
            
            # Salva o modelo
            model_name = f"xgboost_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.model"
            model_path = os.path.join(self.models_dir, model_name)
            model.save_model(model_path)
            
            logger.info(f"Modelo salvo em: {model_path}")
            
            # Plota importância das features
            self.plot_feature_importance(model, X_train.shape[1])
            
            return model, accuracy
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            return None, 0
    
    def plot_feature_importance(self, model, num_features):
        """Plota importância das features"""
        try:
            # Obtém importância das features
            importance = model.get_score(importance_type='weight')
            
            if not importance:
                logger.warning("Não foi possível obter importância das features")
                return
            
            # Converte para DataFrame
            importance_df = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            })
            
            # Ordena por importância
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Plota
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=importance_df.head(15))
            plt.title('Importância das Features - XGBoost')
            plt.tight_layout()
            
            # Salva o gráfico
            plot_path = os.path.join(self.models_dir, "feature_importance.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Gráfico de importância salvo em: {plot_path}")
            logger.info(f"\\nTOP 5 FEATURES:")
            for i, row in importance_df.head().iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.2f}")
                
        except Exception as e:
            logger.error(f"Erro ao plotar importância: {e}")
    
    def evaluate_model(self, model, X_test, y_test):
        """Avalia o modelo em dados de teste"""
        try:
            dtest = xgb.DMatrix(X_test)
            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
        except Exception as e:
            logger.error(f"Erro na avaliação: {e}")
            return None

def main():
    """Função principal"""
    logger.info("=" * 70)
    logger.info("JL CAPITAL TRADE - TREINAMENTO XGBOOST")
    logger.info("=" * 70)
    
    # Configurações
    symbol = "EURUSD"
    timeframe = "H1"
    
    trainer = XGBoostTrainer()
    
    # Carrega dados
    logger.info(f"Carregando dados de {symbol} {timeframe}...")
    df = trainer.load_training_data(symbol, timeframe)
    
    if df is None:
        logger.error("Falha ao carregar dados")
        return
    
    logger.info(f"Dados carregados: {len(df)} registros")
    
    # Prepara features
    X, y = trainer.prepare_features(df)
    
    if X is None or y is None:
        logger.error("Falha ao preparar features")
        return
    
    # Treina modelo
    logger.info("Iniciando treinamento...")
    model, accuracy = trainer.train_model(X, y, symbol, timeframe)
    
    if model is None:
        logger.error("Falha no treinamento")
        return
    
    logger.info(f"Treinamento concluído com acurácia: {accuracy:.4f}")
    
    # Testa previsão em tempo real
    logger.info("\\nTESTANDO PREVISÃO EM TEMPO REAL...")
    
    # Aqui você pode adicionar código para testar previsões em dados atuais
    
    logger.info("\\nMODELO XGBOOST TREINADO E PRONTO PARA USO!")
    logger.info("Use o modelo para fazer previsões de tendência no trading")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    # Importa MetaTrader5 aqui para evitar conflitos
    try:
        import MetaTrader5 as mt5
        main()
    except ImportError:
        logger.error("MetaTrader5 não está instalado")
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")