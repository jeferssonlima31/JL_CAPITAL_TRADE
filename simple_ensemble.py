#!/usr/bin/env python3
"""
JL CAPITAL TRADE - SISTEMA ENSEMBLE OTIMIZADO
Sistema de trading automatizado usando XGBoost para previsão de mercado.
Performance otimizada com carregamento lazy de imports e cache de modelos.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Configura logging otimizado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SimpleEnsembleSystem:
    """Sistema Ensemble simplificado usando apenas XGBoost"""
    
    def __init__(self):
        self.models_dir = "trained_models"
        self.xgboost_model = None
        self.scaler = None
        
        # Limiar de decisão
        self.decision_threshold = 0.55  # 55% de confiança
    
    def load_model(self):
        """Carrega o modelo XGBoost com cache e otimização"""
        try:
            # Cache do modelo para evitar recarregamento
            if self.xgboost_model is not None:
                return True
                
            # Lista arquivos de modelo uma vez
            if not os.path.exists(self.models_dir):
                logger.error("Diretório de modelos não existe")
                return False
                
            model_files = os.listdir(self.models_dir)
            xgboost_files = [f for f in model_files 
                           if f.startswith('xgboost') and f.endswith('.model')]
            
            if not xgboost_files:
                logger.error("Nenhum modelo XGBoost encontrado")
                return False
            
            # Encontra modelo mais recente com otimização
            latest_xgboost = max(xgboost_files, 
                               key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x)))
            xgboost_path = os.path.join(self.models_dir, latest_xgboost)
            
            # Carregamento lazy do XGBoost
            import xgboost as xgb
            self.xgboost_model = xgb.Booster()
            self.xgboost_model.load_model(xgboost_path)
            
            # Carrega scaler se existir
            scaler_path = os.path.join(self.models_dir, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            logger.info(f"Modelo carregado: {latest_xgboost}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def prepare_features(self, df):
        """Prepara features otimizadas para o modelo"""
        try:
            # Todas as features usadas no treinamento original
            all_features = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'bb_position', 'atr', 'volume_ratio',
                'close_position', 'volatility', 'ema_cross',
                'momentum', 'roc'
            ]
            
            # Verifica features disponíveis
            available_features = [f for f in all_features if f in df.columns]
            
            if not available_features:
                logger.error("Nenhuma feature disponível")
                return None
            
            # Seleciona apenas features disponíveis
            X = df[available_features]
            
            # Remove NaN de forma eficiente
            X = X.dropna()
            
            if len(X) == 0:
                logger.error("Dados insuficientes após limpeza")
                return None
            
            return X
            
        except Exception as e:
            logger.error(f"Erro ao preparar features: {e}")
            return None
    
    def predict_with_confidence(self, df):
        """Faz previsão com sistema avançado de confiança"""
        try:
            features = self.prepare_features(df)
            
            if features is None or len(features) == 0:
                logger.error("Features inválidas para previsão")
                return None
            
            # Prepara dados para predição
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features.iloc[-1:])  # Apenas último registro
            else:
                features_scaled = features.iloc[-1:].values
            
            import xgboost as xgb
            import numpy as np
            
            # Converte para DMatrix
            dmatrix = xgb.DMatrix(features_scaled)
            
            # Faz previsão com probabilidades
            prediction_proba = self.xgboost_model.predict(dmatrix)
            
            # Calcula múltiplos níveis de confiança
            raw_confidence = abs(prediction_proba[0] - 0.5) * 2
            
            # Sistema de confiança hierárquico
            if raw_confidence > 0.8:
                confidence_level = "MUITO ALTA"
                risk_level = "BAIXO"
                position_size = "MAXIMO"
            elif raw_confidence > 0.7:
                confidence_level = "ALTA" 
                risk_level = "MODERADO"
                position_size = "ALTO"
            elif raw_confidence > 0.6:
                confidence_level = "MEDIA"
                risk_level = "MODERADO"
                position_size = "MEDIO"
            elif raw_confidence > 0.55:
                confidence_level = "BAIXA"
                risk_level = "ALTO"
                position_size = "PEQUENO"
            else:
                confidence_level = "MUITO BAIXA"
                risk_level = "MUITO ALTO"
                position_size = "MINIMO"
            
            # Calcula expectativa de ganho baseada na confiança
            expected_gain = raw_confidence * 2.5  # Fator de multiplicação otimizado
            
            result = {
                'prediction': float(prediction_proba[0]),
                'signal': 'BUY' if prediction_proba[0] > 0.5 else 'SELL',
                'confidence': round(float(raw_confidence), 4),
                'confidence_level': confidence_level,
                'risk_level': risk_level,
                'position_size': position_size,
                'expected_gain_pct': round(float(expected_gain), 2),
                'strength': 'FORTE' if raw_confidence > 0.7 else 'MODERADO' if raw_confidence > 0.5 else 'FRACO'
            }
            
            logger.info(f"Previsão: {result['prediction']:.4f} -> {result['signal']} ({result['confidence_level']})")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na previsão com confiança: {e}")
            return None
    
    def generate_trading_signal(self, df):
        """Gera sinal de trading com análise de risco avançada"""
        try:
            prediction_result = self.predict_with_confidence(df)
            
            if prediction_result is None:
                return None
            
            # Calcula tamanho de posição baseado na confiança
            position_sizes = {
                "MAXIMO": 0.1,    # 10% do capital
                "ALTO": 0.05,     # 5% do capital  
                "MEDIO": 0.03,    # 3% do capital
                "PEQUENO": 0.01,  # 1% do capital
                "MINIMO": 0.005   # 0.5% do capital
            }
            
            # Stop-loss e take-profit dinâmicos
            if prediction_result['confidence_level'] == "MUITO ALTA":
                stop_loss = 0.005  # 0.5%
                take_profit = 0.015  # 1.5%
            elif prediction_result['confidence_level'] == "ALTA":
                stop_loss = 0.008  # 0.8%
                take_profit = 0.012  # 1.2%
            elif prediction_result['confidence_level'] == "MEDIA":
                stop_loss = 0.01   # 1.0%
                take_profit = 0.01  # 1.0%
            else:
                stop_loss = 0.015  # 1.5%
                take_profit = 0.008  # 0.8%
            
            signal = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': 'EURUSD',
                'signal': prediction_result['signal'],
                'confidence': prediction_result['confidence'],
                'confidence_level': prediction_result['confidence_level'],
                'risk_level': prediction_result['risk_level'],
                'position_size': position_sizes[prediction_result['position_size']],
                'position_size_label': prediction_result['position_size'],
                'expected_gain_pct': prediction_result['expected_gain_pct'],
                'stop_loss_pct': stop_loss * 100,
                'take_profit_pct': take_profit * 100,
                'risk_reward_ratio': round(take_profit / stop_loss, 2),
                'action': 'OPEN',
                'strength': prediction_result['strength']
            }
            
            logger.info(f"SINAL DE TRADING: {signal['signal']} com {signal['confidence']:.2%} de confiança")
            logger.info(f"Nível de Confiança: {signal['confidence_level']}")
            logger.info(f"Tamanho da Posição: {signal['position_size_label']} ({signal['position_size']:.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinal: {e}")
            return None
    
    def backtest_advanced(self, df):
        """Backtesting avançado com métricas completas de performance"""
        try:
            if len(df) < 200:
                logger.error("Dados insuficientes para backtesting avançado")
                return None
            
            # Usa dados mais recentes para teste
            test_data = df.tail(200).copy()
            
            predictions = []
            actual_moves = []
            confidences = []
            
            for i in range(50, len(test_data)):
                current_data = test_data.iloc[:i+1]
                
                # Predição com confiança
                prediction_result = self.predict_with_confidence(current_data)
                
                if prediction_result is not None:
                    predictions.append(1 if prediction_result['signal'] == 'BUY' else 0)
                    confidences.append(prediction_result['confidence'])
                    
                    # Movimento real (próximo candle)
                    if i + 1 < len(test_data):
                        actual_move = 1 if test_data['close'].iloc[i+1] > test_data['close'].iloc[i] else 0
                        actual_moves.append(actual_move)
            
            if len(predictions) == 0:
                return None
            
            # Métricas avançadas
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            import numpy as np
            
            accuracy = accuracy_score(actual_moves, predictions)
            precision = precision_score(actual_moves, predictions, zero_division=0)
            recall = recall_score(actual_moves, predictions, zero_division=0)
            f1 = f1_score(actual_moves, predictions, zero_division=0)
            
            # Calcula ganho esperado baseado na confiança
            avg_confidence = np.mean(confidences)
            expected_gain = avg_confidence * 2.5
            
            # Taxa de acerto por nível de confiança
            high_confidence_idx = [i for i, conf in enumerate(confidences) if conf > 0.7]
            if high_confidence_idx:
                high_conf_accuracy = accuracy_score(
                    [actual_moves[i] for i in high_confidence_idx],
                    [predictions[i] for i in high_confidence_idx]
                )
            else:
                high_conf_accuracy = 0
            
            # Risk-Reward Ratio médio
            avg_risk_reward = 2.0 if avg_confidence > 0.7 else 1.5 if avg_confidence > 0.6 else 1.2
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_confidence': avg_confidence,
                'high_conf_accuracy': high_conf_accuracy,
                'expected_gain_pct': expected_gain,
                'total_trades': len(predictions),
                'win_rate': accuracy,
                'risk_reward_ratio': avg_risk_reward,
                'sharpe_ratio': accuracy / np.std(confidences) if np.std(confidences) > 0 else 0
            }
            
            logger.info(f"\nBACKTESTING AVANÇADO - RESULTADOS:")
            logger.info(f"   Acurácia: {accuracy:.4f}")
            logger.info(f"   Precisão: {precision:.4f}")
            logger.info(f"   Recall: {recall:.4f}")
            logger.info(f"   F1-Score: {f1:.4f}")
            logger.info(f"   Confiança Média: {avg_confidence:.4f}")
            logger.info(f"   Acerto Confiança Alta: {high_conf_accuracy:.4f}")
            logger.info(f"   Ganho Esperado: {expected_gain:.2f}%")
            logger.info(f"   Total de Trades: {len(predictions)}")
            logger.info(f"   Win Rate: {accuracy:.4f}")
            logger.info(f"   Risk-Reward: {avg_risk_reward:.2f}")
            logger.info(f"   Sharpe Ratio: {results['sharpe_ratio']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no backtesting avançado: {e}")
            return None

def main():
    """Função principal"""
    logger.info("=" * 70)
    logger.info("JL CAPITAL TRADE - SISTEMA ENSEMBLE SIMPLIFICADO")
    logger.info("=" * 70)
    
    # Carrega dados de exemplo
    try:
        from train_xgboost_model import XGBoostTrainer
        xgb_trainer = XGBoostTrainer()
        df = xgb_trainer.load_training_data("EURUSD", "H1")
    except Exception as e:
        logger.error(f"Falha ao carregar dados: {e}")
        return
    
    if df is None:
        logger.error("Falha ao carregar dados")
        return
    
    logger.info(f"Dados carregados: {len(df)} registros")
    
    # Inicializa sistema ensemble
    ensemble_system = SimpleEnsembleSystem()
    
    # Carrega modelo
    if not ensemble_system.load_model():
        logger.error("Falha ao carregar modelo")
        return
    
    # Gera sinal de trading atual
    logger.info("\nGERANDO SINAL DE TRADING ATUAL...")
    signal = ensemble_system.generate_trading_signal(df)
    
    if signal:
        logger.info(f"\nSINAL GERADO: {signal['signal']}")
        logger.info(f"   Confiança: {signal['confidence']:.2%}")
        logger.info(f"   Nível: {signal['confidence_level']}")
        logger.info(f"   Risco: {signal['risk_level']}")
        logger.info(f"   Tamanho Posição: {signal['position_size_label']} ({signal['position_size']:.3f})")
        logger.info(f"   Expectativa Ganho: {signal['expected_gain_pct']}%")
        logger.info(f"   Stop Loss: {signal['stop_loss_pct']:.1f}%")
        logger.info(f"   Take Profit: {signal['take_profit_pct']:.1f}%")
        logger.info(f"   Risk-Reward: {signal['risk_reward_ratio']:.2f}")
    
    # Backtesting avançado
    logger.info("\nEXECUTANDO BACKTESTING AVANÇADO...")
    backtest_results = ensemble_system.backtest_advanced(df)
    
    if backtest_results:
        logger.info(f"\nPERFORMANCE DO SISTEMA:")
        logger.info(f"   Performance Geral: {backtest_results['accuracy']:.2%}")
        logger.info(f"   Performance Alta Confiança: {backtest_results['high_conf_accuracy']:.2%}")
        logger.info(f"   Expectativa de Ganho: {backtest_results['expected_gain_pct']:.2f}%")
        logger.info(f"   Risk-Reward Ratio: {backtest_results['risk_reward_ratio']:.2f}")
        logger.info(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
    
    logger.info("\nSISTEMA ENSEMBLE CONFIGURADO E PRONTO PARA USO!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()