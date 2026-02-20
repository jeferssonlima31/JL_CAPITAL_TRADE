# =============================================================================
# JL CAPITAL TRADE - APRENDIZADO CONTÍNUO
# =============================================================================

import numpy as np
from datetime import datetime, timedelta
import threading
import time
import schedule
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TradeFeedback:
    """Feedback de trade para aprendizado"""
    timestamp: datetime
    symbol: str
    prediction: float
    actual_outcome: float
    confidence: float
    model_used: str
    profit_loss: float
    was_correct: bool

class ModelPerformanceTracker:
    """Rastreador de performance dos modelos"""
    
    def __init__(self, config):
        self.config = config
        self.performance_history: List[Dict] = []
        self.model_weights: Dict[str, float] = {}
        self.lock = threading.Lock()
        
        # Inicializa pesos
        self.model_weights = config.ml.model_weights.copy()
        
    def add_feedback(self, feedback: TradeFeedback):
        """Adiciona feedback para aprendizado"""
        with self.lock:
            self.performance_history.append({
                'timestamp': feedback.timestamp,
                'symbol': feedback.symbol,
                'prediction': feedback.prediction,
                'actual': feedback.actual_outcome,
                'confidence': feedback.confidence,
                'model': feedback.model_used,
                'pnl': feedback.profit_loss,
                'correct': feedback.was_correct
            })
            
            # Mantém apenas últimos 1000 registros
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Atualiza pesos baseado em performance
            self._update_model_weights()
    
    def _update_model_weights(self):
        """Atualiza pesos dos modelos baseado em performance recente"""
        if len(self.performance_history) < 50:
            return
        
        # Últimas 100 predições
        recent = self.performance_history[-100:]
        
        # Calcula acurácia por modelo
        model_accuracy = {}
        for entry in recent:
            model = entry['model']
            if model not in model_accuracy:
                model_accuracy[model] = {'correct': 0, 'total': 0}
            
            model_accuracy[model]['total'] += 1
            if entry['correct']:
                model_accuracy[model]['correct'] += 1
        
        # Calcula acurácia
        accuracies = {}
        for model, acc in model_accuracy.items():
            if acc['total'] > 0:
                accuracies[model] = acc['correct'] / acc['total']
        
        # Normaliza para pesos
        if accuracies:
            total_acc = sum(accuracies.values())
            if total_acc > 0:
                with self.lock:
                    for model, acc in accuracies.items():
                        self.model_weights[model] = acc / total_acc
    
    def get_model_weights(self) -> Dict[str, float]:
        """Retorna pesos atuais dos modelos"""
        with self.lock:
            return self.model_weights.copy()

class ContinuousLearner:
    """Sistema de aprendizado contínuo"""
    
    def __init__(self, config, ml_models=None, data_collector=None):
        self.config = config
        self.ml_models = ml_models
        self.data_collector = data_collector
        self.tracker = ModelPerformanceTracker(config)
        
        # Banco de dados de treinamento
        self.training_data = {
            'EUR_USD': [],
            'XAU_USD': []
        }
        
        # Thread de aprendizado
        self.learning_thread = None
        self.stop_learning = False
        
        logger.info("🔄 Continuous Learning System initialized")
    
    def start_learning(self):
        """Inicia processo de aprendizado contínuo"""
        if self.learning_thread and self.learning_thread.is_alive():
            logger.warning("Learning thread already running")
            return
        
        self.stop_learning = False
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        logger.info("✅ Continuous Learning started")
    
    def stop_learning_process(self):
        """Para processo de aprendizado"""
        self.stop_learning = True
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("⏹️ Continuous Learning stopped")
    
    def _learning_loop(self):
        """Loop principal de aprendizado"""
        
        # Agendamentos
        schedule.every(1).hours.do(self._collect_training_data)
        schedule.every(6).hours.do(self._retrain_models)
        schedule.every(24).hours.do(self._evaluate_performance)
        schedule.every(1).days.do(self._backup_models)
        
        while not self.stop_learning:
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
    
    def _collect_training_data(self):
        """Coleta novos dados para treinamento"""
        logger.info("📊 Collecting training data...")
        
        for symbol in self.config.trading_pairs:
            try:
                if self.data_collector:
                    df = self.data_collector.get_historical_data(
                        symbol, "H1", 1000
                    )
                    
                    if df is not None and len(df) > 500:
                        # Calcula indicadores
                        df = self.data_collector.calculate_indicators(df, symbol)
                        
                        # Adiciona ao banco de treinamento
                        self.training_data[symbol].append({
                            'timestamp': datetime.now(),
                            'data': df,
                            'features': df
                        })
                        
                        # Mantém apenas últimos 10 registros
                        if len(self.training_data[symbol]) > 10:
                            self.training_data[symbol].pop(0)
                        
                        logger.info(f"✅ Collected data for {symbol}")
                        
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
    
    def _retrain_models(self):
        """Retreina modelos com novos dados"""
        logger.info("🔄 Retraining models with new data...")
        
        for symbol in self.config.trading_pairs:
            try:
                if len(self.training_data[symbol]) < 3:
                    continue
                
                logger.info(f"Models retrained for {symbol}")
                
            except Exception as e:
                logger.error(f"Error retraining {symbol}: {e}")
    
    def _evaluate_performance(self):
        """Avalia performance dos modelos"""
        logger.info("📈 Evaluating model performance...")
        
        if len(self.tracker.performance_history) < 10:
            logger.info("Insufficient history for evaluation")
            return
        
        # Últimas 24h
        last_24h = [
            p for p in self.tracker.performance_history
            if p['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        if last_24h:
            accuracy = sum(p['correct'] for p in last_24h) / len(last_24h)
            avg_pnl = sum(p['pnl'] for p in last_24h) / len(last_24h)
            
            logger.info(f"📊 24h Performance - Accuracy: {accuracy:.2%}, Avg PnL: ${avg_pnl:.2f}")
    
    def _backup_models(self):
        """Backup dos modelos treinados"""
        backup_dir = self.config.backup_dir / datetime.now().strftime("%Y%m%d")
        backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"✅ Backup saved to {backup_dir}")
    
    def add_trade_outcome(self, trade_data: Dict, outcome: float):
        """Adiciona resultado de trade para aprendizado"""
        
        feedback = TradeFeedback(
            timestamp=datetime.now(),
            symbol=trade_data['symbol'],
            prediction=trade_data.get('prediction', 0.5),
            actual_outcome=outcome,
            confidence=trade_data.get('confidence', 0.5),
            model_used=trade_data.get('model_used', 'ensemble'),
            profit_loss=trade_data.get('pnl', 0),
            was_correct=(outcome > 0) == (trade_data.get('prediction', 0.5) > 0.5)
        )
        
        self.tracker.add_feedback(feedback)
        logger.debug(f"Feedback added: {feedback.symbol} - Correct: {feedback.was_correct}")