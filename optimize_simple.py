#!/usr/bin/env python3
# =============================================================================
# JL CAPITAL TRADE - OTIMIZAÇÃO SIMPLES E RÁPIDA
# =============================================================================

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_simple():
    """Otimização simples e rápida"""
    try:
        # Carrega dados
        data_path = "trained_models/training_data_enhanced.csv"
        df = pd.read_csv(data_path)
        
        # Separa features e target
        X = df.drop(['target', 'symbol'], axis=1, errors='ignore')
        y = df['target']
        
        # Remove NaN
        X = X.fillna(X.median())
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Dados: {X.shape[0]} registros, {X.shape[1]} features")
        logger.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
        
        # Testa algumas combinações manuais (rápido)
        best_accuracy = 0
        best_params = {}
        
        # Combinações testadas
        combinations = [
            {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200},
            {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 300},
            {'max_depth': 3, 'learning_rate': 0.2, 'n_estimators': 100},
            {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 150, 'subsample': 0.9},
            {'max_depth': 7, 'learning_rate': 0.08, 'n_estimators': 250, 'colsample_bytree': 0.8}
        ]
        
        for i, params in enumerate(combinations):
            logger.info(f"Testando combinação {i+1}/{len(combinations)}")
            
            model = xgb.XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                best_model = model
            
            logger.info(f"  Acurácia: {accuracy:.4f}")
        
        # Resultados finais
        logger.info(f"\n🎯 MELHOR COMBINAÇÃO:")
        logger.info(f"Parâmetros: {best_params}")
        logger.info(f"Acurácia: {best_accuracy:.4f}")
        
        # Relatório completo
        y_pred_final = best_model.predict(X_test)
        logger.info("\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
        logger.info(classification_report(y_test, y_pred_final))
        
        # Treina modelo final com todos os dados
        final_model = xgb.XGBClassifier(
            **best_params,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        final_model.fit(X, y)
        
        # Salva modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"trained_models/xgboost_optimized_{timestamp}.model"
        final_model.save_model(model_filename)
        
        # Salva parâmetros
        params_df = pd.Series(best_params)
        params_df.to_json(f"trained_models/best_params_{timestamp}.json")
        
        logger.info(f"\n✅ MODELO SALVO: {model_filename}")
        logger.info(f"📊 ACURÁCIA FINAL: {best_accuracy:.2%}")
        
        return best_accuracy
        
    except Exception as e:
        logger.error(f"Erro na otimização: {e}")
        return None

def main():
    """Função principal"""
    logger.info("=" * 60)
    logger.info("JL CAPITAL TRADE - OTIMIZAÇÃO RÁPIDA")
    logger.info("=" * 60)
    
    accuracy = optimize_simple()
    
    if accuracy:
        logger.info(f"\n🚀 OTIMIZAÇÃO CONCLUÍDA!")
        logger.info(f"🎯 Acurácia alcançada: {accuracy:.2%}")
    else:
        logger.error("❌ Falha na otimização")

if __name__ == "__main__":
    main()