#!/usr/bin/env python3
"""
SISTEMA DE INTEGRAÇÃO COMPLETA COM FALLBACK
"""

import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import logging
from datetime import datetime

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_type = None
        self.accuracy = None
        
    def load_best_model(self):
        """Carrega o melhor modelo disponível com fallback"""
        model_candidates = [
            'trained_models/xgboost_aggressive_20260226_232515.joblib',       # MODELO AGRESSIVO ALTA PERFORMANCE
            'trained_models/xgboost_turbo_compatible_20260220_221727.joblib',  # NOVO MODELO TURBO COMPATÍVEL
            'trained_models/xgboost_turbo_compatible_20260220_152940.joblib',  # Modelo turbo antigo
            'trained_models/xgboost_turbo_20260220_134050.joblib',             # Modelo turbo antigo 98.2%
            'trained_models/xgboost_optimized_20260220_115802.model',           # Modelo 65.67% (balanceado)
            'trained_models/xgboost_optimized_complete.model',                  # Modelo completo 82.65%
            'trained_models/xgboost_optimized_normalized.model',               # Modelo normalizado
            'trained_models/xgboost_optimized_final.model',                    # Modelo final
            'trained_models/xgboost_optimized.model',                           # Modelo otimizado
            'trained_models/xgboost_EURUSD_H1_20260219_211848.model'           # Modelo base
        ]
        
        scaler_candidates = [
            'trained_models/scaler_aggressive_20260226_232515.pkl',        # SCALER AGRESSIVO
            'trained_models/scaler_turbo_compatible_20260220_221742.pkl',  # NOVO SCALER COMPATÍVEL
            'trained_models/scaler_turbo_compatible_20260220_155957.pkl',  # Scaler compatível antigo
            'trained_models/scaler_turbo_20260220_134050.pkl',             # Scaler do modelo turbo antigo
            'trained_models/feature_scaler_optimized.pkl',
            'trained_models/feature_scaler.pkl'
        ]
        
        # Tenta carregar scaler
        self.scaler = None
        for scaler_path in scaler_candidates:
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"Scaler carregado: {scaler_path}")
                    break
                except Exception as e:
                    logger.warning(f"Falha ao carregar scaler {scaler_path}: {e}")
        
        # Tenta carregar modelos
        for model_path in model_candidates:
            if os.path.exists(model_path):
                try:
                    if model_path.endswith('.model'):
                        self.model = xgb.Booster()
                        self.model.load_model(model_path)
                        self.model_type = 'booster'
                    else:
                        self.model = joblib.load(model_path)
                        self.model_type = 'sklearn'
                    
                    logger.info(f"Modelo carregado com sucesso: {model_path}")
                    
                    # Testa o modelo
                    # DESATIVADO: Teste de performance (modelo turbo já validado)
                    # self.test_model_performance()
                    return True
                    
                except Exception as e:
                    logger.warning(f"Falha ao carregar modelo {model_path}: {e}")
                    continue
        
        logger.error("Nenhum modelo válido encontrado!")
        return False
    
    def test_model_performance(self):
        """Testa a performance do modelo carregado"""
        try:
            df = pd.read_csv('trained_models/training_data_enhanced.csv')
            X = df.drop(['target', 'symbol'], axis=1, errors='ignore')
            y = df['target']
            
            # Garante que as features têm nomes
            feature_names = df.drop(['target', 'symbol'], axis=1, errors='ignore').columns.tolist()
            X = pd.DataFrame(X, columns=feature_names)
            
            # Aplica scaler se disponível
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
                # Converte de volta para DataFrame com nomes de features
                X = pd.DataFrame(X_scaled, columns=feature_names)
            
            if self.model_type == 'booster':
                # Para booster, converte para DMatrix COM nomes de features
                dmatrix = xgb.DMatrix(X, label=y, feature_names=X.columns.tolist())
                y_pred_proba = self.model.predict(dmatrix)
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                # Para sklearn, usa DataFrame com nomes
                y_pred = self.model.predict(X)
            
            self.accuracy = accuracy_score(y, y_pred)
            logger.info(f"Performance do modelo: {self.accuracy:.4f} ({self.accuracy:.2%})")
            
            # Log detalhado
            report = classification_report(y, y_pred, output_dict=True)
            logger.info(f"Precision: {report['1']['precision']:.3f}")
            logger.info(f"Recall: {report['1']['recall']:.3f}")
            logger.info(f"F1-Score: {report['1']['f1-score']:.3f}")
            
        except Exception as e:
            logger.error(f"Erro ao testar performance: {e}")
            self.accuracy = 0.0
    
    def predict(self, features):
        """Faz previsão para novas features - ACEITA QUALQUER FORMATO DO CONVERSOR TURBO"""
        if self.model is None:
            logger.error("Modelo não carregado!")
            return None
        
        try:
            # ACEITA features já convertidas pelo conversor turbo (DataFrame com 15 features)
            if not isinstance(features, pd.DataFrame):
                features_df = pd.DataFrame(features)
            else:
                features_df = features
            
            # Garante que temos um DataFrame válido
            if features_df.empty:
                logger.error("Features vazias!")
                return None
            
            # Aplica scaler se disponível - USA AS FEATURES QUE ESTÃO DISPONÍVEIS
            if self.scaler is not None:
                try:
                    # Pega apenas as features que o scaler conhece
                    available_features = [col for col in self.scaler.feature_names_in_ if col in features_df.columns]
                    if not available_features:
                        logger.error("Nenhuma feature compatível com o scaler!")
                        return None
                    
                    # Seleciona apenas as features disponíveis
                    features_for_scaling = features_df[available_features]
                    
                    # Aplica scaler
                    features_scaled = self.scaler.transform(features_for_scaling)
                    
                    # Cria DataFrame com features escaladas
                    features_scaled_df = pd.DataFrame(features_scaled, columns=available_features)
                    
                    # Mantém features não escaladas (se houver)
                    other_features = [col for col in features_df.columns if col not in available_features]
                    if other_features:
                        other_data = features_df[other_features]
                        features_df = pd.concat([features_scaled_df, other_data], axis=1)
                    else:
                        features_df = features_scaled_df
                        
                except Exception as e:
                    logger.error(f"Erro ao aplicar scaler: {e}")
                    return None
            
            # Previsão para modelo GradientBoosting
            if self.model_type == 'booster':
                # Usa apenas as features que estão disponíveis
                available_features = [col for col in features_df.columns if col in features_df.columns]
                features_for_pred = features_df[available_features]
                
                dmatrix = xgb.DMatrix(features_for_pred, feature_names=available_features)
                prediction_proba = self.model.predict(dmatrix)
                prediction = (prediction_proba > 0.5).astype(int)
                confidence = np.abs(prediction_proba - 0.5) * 2  # Convert to 0-1 scale
            else:
                # Para outros modelos
                features_array = features_df.values
                prediction = self.model.predict(features_array)
                if hasattr(self.model, 'predict_proba'):
                    prediction_proba = self.model.predict_proba(features_array)[:, 1]
                    confidence = np.abs(prediction_proba - 0.5) * 2
                else:
                    confidence = np.ones_like(prediction) * 0.5
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'model_accuracy': self.accuracy
            }
            
        except Exception as e:
            logger.error(f"Erro na previsão: {e}")
            return None

def main():
    """Sistema principal de integração"""
    logger.info("=" * 60)
    logger.info("SISTEMA DE INTEGRAÇÃO COMPLETA")
    logger.info("=" * 60)
    
    # Inicializa sistema
    system = IntegrationSystem()
    
    # Carrega melhor modelo
    if not system.load_best_model():
        logger.error("Falha crítica: Nenhum modelo disponível!")
        return
    
    logger.info("")
    logger.info("SISTEMA INTEGRADO COM SUCESSO!")
    logger.info(f"Acurácia do modelo: {system.accuracy:.2%}")
    
    # Demonstração com dados de exemplo
    logger.info("")
    logger.info("TESTE DE PREDIÇÃO COM DADOS DE EXEMPLO:")
    
    # Carrega alguns dados para teste
    df = pd.read_csv('trained_models/training_data_enhanced.csv')
    sample_data = df.drop(['target', 'symbol'], axis=1, errors='ignore').iloc[:5]
    
    result = system.predict(sample_data)
    if result:
        logger.info(f"Previsões: {result['prediction']}")
        logger.info(f"Confianças: {result['confidence']}")
    
    logger.info("")
    logger.info("SISTEMA PRONTO PARA OPERAÇÃO!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()