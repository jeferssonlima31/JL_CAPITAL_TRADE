#!/usr/bin/env python3
"""
BACKTESTING AVANÇADO DO SISTEMA COMPLETO
"""

import pandas as pd
import numpy as np
from integration_system import IntegrationSystem
import logging

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtesting():
    """Executa backtesting completo do sistema"""
    
    logger.info("=" * 60)
    logger.info("BACKTESTING AVANÇADO - JL CAPITAL TRADE")
    logger.info("=" * 60)
    
    # Carrega sistema integrado
    system = IntegrationSystem()
    if not system.load_best_model():
        logger.error("Falha ao carregar modelo!")
        return
    
    # Carrega dados para backtesting
    logger.info("Carregando dados para backtesting...")
    df = pd.read_csv('trained_models/training_data_enhanced.csv')
    
    # Usa os últimos 1000 registros para backtesting
    test_data = df.tail(1000).copy()
    X_test = test_data.drop(['target', 'symbol'], axis=1, errors='ignore')
    y_true = test_data['target']
    
    logger.info(f"Backtesting com {len(test_data)} registros...")
    
    # Faz previsões
    predictions = []
    confidences = []
    
    for i in range(len(test_data)):
        if i % 100 == 0:
            logger.info(f"Processando {i}/{len(test_data)}...")
        
        # Pega features para este registro
        features = X_test.iloc[i:i+1]
        
        # Faz previsão
        result = system.predict(features)
        if result:
            predictions.append(result['prediction'][0])
            confidences.append(result['confidence'][0])
        else:
            predictions.append(0)
            confidences.append(0.5)
    
    # Calcula métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    
    # Estatísticas de confiança
    avg_confidence = np.mean(confidences)
    confidence_std = np.std(confidences)
    
    logger.info("=" * 60)
    logger.info("RESULTADOS DO BACKTESTING:")
    logger.info("=" * 60)
    logger.info(f"Acurácia: {accuracy:.4f} ({accuracy:.2%})")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"Confiança média: {avg_confidence:.4f} ({avg_confidence:.2%})")
    logger.info(f"Desvio padrão da confiança: {confidence_std:.4f}")
    
    # Análise de distribuição
    logger.info("-" * 40)
    logger.info("DISTRIBUIÇÃO DAS PREVISÕES:")
    logger.info(f"Previsões positivas (compra): {sum(predictions)} / {len(predictions)}")
    logger.info(f"Targets positivos reais: {sum(y_true)} / {len(y_true)}")
    
    # Retorna resultados
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_confidence': avg_confidence,
        'confidence_std': confidence_std,
        'predictions': predictions,
        'confidences': confidences
    }

if __name__ == "__main__":
    run_backtesting()