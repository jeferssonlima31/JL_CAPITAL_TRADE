#!/usr/bin/env python3
# =============================================================================
# JL CAPITAL TRADE - TREINAMENTO SIMPLIFICADO DO MODELO LSTM
# =============================================================================

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_lstm_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleLSTMTrainer:
    """Treinador simplificado do modelo LSTM usando MLP"""
    
    def __init__(self):
        self.models_dir = "trained_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Hiperparâmetros para MLP (simulando LSTM)
        self.hidden_layer_sizes = (100, 50, 25)  # Arquitetura profunda
        self.max_iter = 1000
        self.random_state = 42
        
        # Features para o modelo
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position', 'atr', 'volume_ratio',
            'close_position', 'volatility', 'ema_cross',
            'momentum', 'roc', 'returns'
        ]
    
    def create_sequences(self, data, sequence_length=10):
        """Cria sequências para treinamento"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            # Sequência de features (últimos 'sequence_length' períodos)
            seq = data[i:i + sequence_length].flatten()  # Achata a sequência
            
            # Target (próximo período)
            target = data[i + sequence_length, -1]  # última coluna é o target
            
            X.append(seq)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def build_mlp_model(self):
        """Constrói modelo MLP para substituir LSTM"""
        model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=50,
            validation_fraction=0.2
        )
        return model
    
    def prepare_sequence_data(self, df, sequence_length=10):
        """Prepara dados sequenciais"""
        try:
            # Seleciona features disponíveis
            available_features = [f for f in self.feature_columns if f in df.columns]
            
            if not available_features:
                logger.error("Nenhuma feature disponível")
                return None, None, None, None
            
            # Extrai features e target
            X_data = df[available_features].copy()
            y_data = df['target'].copy()
            
            # Remove NaN values
            valid_indices = X_data.notnull().all(axis=1) & y_data.notnull()
            X_data = X_data[valid_indices]
            y_data = y_data[valid_indices]
            
            if len(X_data) < sequence_length * 2:
                logger.error(f"Dados insuficientes para sequências. Necessário: {sequence_length * 2}, Disponível: {len(X_data)}")
                return None, None, None, None
            
            # Normaliza features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            
            # Adiciona target como última coluna
            data_with_target = np.column_stack((X_scaled, y_data))
            
            # Cria sequências
            X_seq, y_seq = self.create_sequences(data_with_target, sequence_length)
            
            logger.info(f"Sequências criadas: {X_seq.shape}")
            logger.info(f"Targets: {y_seq.shape}")
            
            # Divide em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
            )
            
            logger.info(f"Dados de treino: {X_train.shape}")
            logger.info(f"Dados de teste: {X_test.shape}")
            
            # Salva o scaler
            scaler_path = os.path.join(self.models_dir, "simple_lstm_feature_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {e}")
            return None, None, None, None
    
    def train_model(self, X_train, X_test, y_train, y_test, symbol="EURUSD", timeframe="H1"):
        """Treina o modelo MLP"""
        try:
            # Constrói o modelo
            model = self.build_mlp_model()
            
            logger.info("Iniciando treinamento do MLP (simulação LSTM)...")
            
            # Treina o modelo
            model.fit(X_train, y_train)
            
            # Faz previsões
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calcula métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logger.info(f"\nRESULTADOS DO TREINAMENTO MLP:")
            logger.info(f"   Acurácia: {accuracy:.4f}")
            logger.info(f"   Precisão: {precision:.4f}")
            logger.info(f"   Recall: {recall:.4f}")
            logger.info(f"   F1-Score: {f1:.4f}")
            
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"\nMatriz de Confusão MLP:")
            logger.info(f"   Verdadeiros Negativos: {cm[0, 0]}")
            logger.info(f"   Falsos Positivos: {cm[0, 1]}")
            logger.info(f"   Falsos Negativos: {cm[1, 0]}")
            logger.info(f"   Verdadeiros Positivos: {cm[1, 1]}")
            
            # Salva o modelo
            model_name = f"simple_lstm_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = os.path.join(self.models_dir, model_name)
            joblib.dump(model, model_path)
            
            logger.info(f"Modelo MLP salvo em: {model_path}")
            
            # Plota importância das features (simplificado)
            self.plot_feature_importance(model, X_train.shape[1], len(self.feature_columns))
            
            return model, accuracy
            
        except Exception as e:
            logger.error(f"Erro no treinamento MLP: {e}")
            return None, 0
    
    def plot_feature_importance(self, model, total_features, original_features_count):
        """Plota importância das features de forma simplificada"""
        try:
            # Para MLP, usamos os pesos da primeira camada como proxy de importância
            if hasattr(model, 'coefs_') and model.coefs_:
                # Soma absoluta dos pesos para cada feature de entrada
                feature_importance = np.abs(model.coefs_[0]).sum(axis=1)
                
                # Como temos sequências, agrupa as importâncias por feature original
                seq_length = total_features // original_features_count
                
                importance_by_feature = []
                for i in range(original_features_count):
                    start_idx = i * seq_length
                    end_idx = (i + 1) * seq_length
                    importance = feature_importance[start_idx:end_idx].sum()
                    importance_by_feature.append(importance)
                
                # Cria DataFrame
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns[:len(importance_by_feature)],
                    'importance': importance_by_feature
                })
                
                # Ordena por importância
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                # Plota
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=importance_df.head(10))
                plt.title('Importância das Features - MLP (Proxy LSTM)')
                plt.tight_layout()
                
                # Salva o gráfico
                plot_path = os.path.join(self.models_dir, "simple_lstm_feature_importance.png")
                plt.savefig(plot_path)
                plt.close()
                
                logger.info(f"Gráfico de importância salvo em: {plot_path}")
                logger.info(f"\nTOP 5 FEATURES:")
                for i, row in importance_df.head().iterrows():
                    logger.info(f"   {row['feature']}: {row['importance']:.2f}")
            
            else:
                logger.warning("Não foi possível calcular importância das features")
                
        except Exception as e:
            logger.error(f"Erro ao plotar importância: {e}")

def main():
    """Função principal"""
    logger.info("=" * 70)
    logger.info("JL CAPITAL TRADE - TREINAMENTO SIMPLIFICADO LSTM")
    logger.info("=" * 70)
    
    # Configurações
    symbol = "EURUSD"
    timeframe = "H1"
    sequence_length = 10  # 10 períodos históricos
    
    trainer = SimpleLSTMTrainer()
    
    # Carrega dados (reutiliza função do XGBoost)
    try:
        from train_xgboost_model import XGBoostTrainer
        xgb_trainer = XGBoostTrainer()
        df = xgb_trainer.load_training_data(symbol, timeframe)
    except Exception as e:
        logger.error(f"Falha ao carregar dados: {e}")
        return
    
    if df is None:
        logger.error("Falha ao carregar dados")
        return
    
    logger.info(f"Dados carregados: {len(df)} registros")
    
    # Prepara dados sequenciais
    X_train, X_test, y_train, y_test = trainer.prepare_sequence_data(df, sequence_length)
    
    if X_train is None:
        logger.error("Falha ao preparar dados sequenciais")
        return
    
    # Treina modelo
    logger.info("Iniciando treinamento do MLP...")
    model, accuracy = trainer.train_model(X_train, X_test, y_train, y_test, symbol, timeframe)
    
    if model is None:
        logger.error("Falha no treinamento")
        return
    
    logger.info(f"Treinamento concluído com acurácia: {accuracy:.4f}")
    
    logger.info("\nMODELO MLP (SIMULAÇÃO LSTM) TREINADO E PRONTO PARA USO!")
    logger.info("Use o modelo para análise temporal de séries financeiras")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    main()