#!/usr/bin/env python3
# =============================================================================
# JL CAPITAL TRADE - OTIMIZAÇÃO RÁPIDA XGBOOST
# =============================================================================

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostOptimizerFast:
    """Otimizador rápido de hiperparâmetros para XGBoost"""
    
    def __init__(self):
        self.models_dir = "trained_models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_training_data(self):
        """Carrega dados de treinamento"""
        try:
            data_path = os.path.join(self.models_dir, "training_data_enhanced.csv")
            if not os.path.exists(data_path):
                logger.error("Arquivo de dados de treinamento não encontrado")
                return None
            
            df = pd.read_csv(data_path)
            logger.info(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} features")
            
            # Separa features e target
            X = df.drop(['target', 'timestamp', 'symbol'], axis=1, errors='ignore')
            y = df['target']
            
            # Remove colunas com muitos NaN
            X = X.dropna(axis=1, how='all')
            
            # Preenche valores faltantes
            X = X.fillna(X.median())
            
            return X, y
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return None, None
    
    def optimize_hyperparameters(self):
        """Otimiza hiperparâmetros usando RandomizedSearch (mais rápido)"""
        try:
            X, y = self.load_training_data()
            if X is None:
                return None
            
            # Split dos dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
            
            # Hiperparâmetros para otimização (versão reduzida)
            param_dist = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'n_estimators': [100, 200],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'gamma': [0, 0.1],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 1.5]
            }
            
            # Modelo base
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )
            
            # Randomized Search (mais rápido que GridSearch)
            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_dist,
                n_iter=50,  # Apenas 50 combinações aleatórias
                scoring='accuracy',
                cv=3,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            logger.info("Iniciando otimização rápida de hiperparâmetros...")
            random_search.fit(X_train, y_train)
            
            # Melhores parâmetros
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            logger.info(f"Melhores parâmetros: {best_params}")
            logger.info(f"Melhor score (validação): {best_score:.4f}")
            
            # Avaliação no teste
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Acurácia no teste: {test_accuracy:.4f}")
            logger.info("\nRelatório de classificação:")
            logger.info(classification_report(y_test, y_pred))
            
            # Salva o modelo otimizado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"xgboost_optimized_{timestamp}.model"
            model_path = os.path.join(self.models_dir, model_filename)
            
            best_model.save_model(model_path)
            
            # Salva os melhores parâmetros
            params_path = os.path.join(self.models_dir, "best_params.json")
            pd.Series(best_params).to_json(params_path)
            
            logger.info(f"Modelo otimizado salvo: {model_filename}")
            
            return best_params, test_accuracy
            
        except Exception as e:
            logger.error(f"Erro na otimização: {e}")
            return None
    
    def train_final_model(self, best_params):
        """Treina modelo final com os melhores parâmetros"""
        try:
            X, y = self.load_training_data()
            if X is None:
                return None
            
            # Modelo final com melhores parâmetros
            final_model = xgb.XGBClassifier(
                **best_params,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )
            
            # Treina com todos os dados
            final_model.fit(X, y)
            
            # Salva modelo final
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"xgboost_final_{timestamp}.model"
            model_path = os.path.join(self.models_dir, model_filename)
            
            final_model.save_model(model_path)
            
            logger.info(f"Modelo final treinado e salvo: {model_filename}")
            logger.info(f"Total de dados: {X.shape[0]} registros")
            
            return final_model
            
        except Exception as e:
            logger.error(f"Erro no treinamento final: {e}")
            return None

def main():
    """Função principal"""
    logger.info("=" * 60)
    logger.info("JL CAPITAL TRADE - OTIMIZAÇÃO RÁPIDA XGBOOST")
    logger.info("=" * 60)
    
    optimizer = XGBoostOptimizerFast()
    
    # Otimiza hiperparâmetros
    best_params, test_accuracy = optimizer.optimize_hyperparameters()
    
    if best_params:
        logger.info("\n" + "=" * 40)
        logger.info("TREINAMENTO DO MODELO FINAL")
        logger.info("=" * 40)
        
        # Treina modelo final com todos os dados
        final_model = optimizer.train_final_model(best_params)
        
        if final_model:
            logger.info("\n✅ OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
            logger.info(f"📊 Acurácia esperada: {test_accuracy:.2%}")
        else:
            logger.error("❌ Falha no treinamento final")
    else:
        logger.error("❌ Falha na otimização")

if __name__ == "__main__":
    main()