#!/usr/bin/env python3
# =============================================================================
# JL CAPITAL TRADE - CRIA MODELO OTIMIZADO MANUALMENTE
# =============================================================================

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_optimized_model():
    """Cria modelo XGBoost otimizado manualmente"""
    try:
        # Carrega dados
        data_path = "trained_models/training_data_enhanced.csv"
        logger.info(f"Carregando dados de: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} features")
        
        # Separa features e target
        X = df.drop(['target', 'symbol'], axis=1, errors='ignore')
        y = df['target']
        
        # Remove NaN
        X = X.fillna(X.median())
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
        
        # Parâmetros otimizados manualmente (baseado em testes anteriores)
        best_params = {
            'max_depth': 7,
            'learning_rate': 0.08, 
            'n_estimators': 250,
            'colsample_bytree': 0.8,
            'subsample': 0.9,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.5
        }
        
        # Cria e treina modelo
        model = xgb.XGBClassifier(
            **best_params,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        logger.info("Treinando modelo com parâmetros otimizados...")
        model.fit(X_train, y_train)
        
        # Avaliação
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Acurácia do modelo: {accuracy:.4f}")
        
        # Treina com todos os dados
        logger.info("Treinando modelo final com todos os dados...")
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X, y)
        
        # Salva modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"trained_models/xgboost_optimized_{timestamp}.model"
        final_model.save_model(model_filename)
        
        logger.info(f"Modelo salvo: {model_filename}")
        logger.info(f"Acurácia final: {accuracy:.2%}")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Erro ao criar modelo: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Função principal"""
    logger.info("=" * 60)
    logger.info("CRIANDO MODELO XGBOOST OTIMIZADO")
    logger.info("=" * 60)
    
    accuracy = create_optimized_model()
    
    if accuracy:
        logger.info(f"\nMODELO CRIADO COM SUCESSO!")
        logger.info(f"Acurácia: {accuracy:.2%}")
    else:
        logger.error("Falha ao criar modelo")

if __name__ == "__main__":
    main()