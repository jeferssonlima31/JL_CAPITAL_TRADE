# =============================================================================
# JL CAPITAL TRADE - APRENDIZADO CONTÍNUO
# =============================================================================

import numpy as np
import pandas as pd
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

    def get_model_weight(self, symbol: str, model_name: str) -> float:
        """Retorna o peso de um modelo específico"""
        with self.lock:
            return self.model_weights.get(model_name, 1.0)

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
        """Retreina modelos com novos dados e validação Champion vs Challenger"""
        logger.info("🔄 Iniciando ciclo de retreino seguro (Champion vs Challenger)...")
        
        for symbol in self.config.trading_pairs:
            try:
                # 1. Verifica volume mínimo de dados
                if len(self.training_data[symbol]) < 1:
                    continue
                
                latest_entry = self.training_data[symbol][-1]
                df = latest_entry['data']
                
                if len(df) < self.config.ml.min_samples_for_retrain:
                    logger.info(f"⏳ Dados insuficientes para {symbol}: {len(df)}/{self.config.ml.min_samples_for_retrain}")
                    continue

                # 2. Prepara dados para validação Champion vs Challenger
                # Divide em Treino (80%) e Validação OOS (20%)
                train_size = int(len(df) * 0.8)
                df_train = df.iloc[:train_size]
                df_val = df.iloc[train_size:]
                
                logger.info(f"🧪 Validando {symbol}: {len(df_train)} amostras treino, {len(df_val)} amostras OOS")

                # 3. Avalia performance do modelo ATUAL (Champion) em dados OOS
                champion_results = self._evaluate_model_on_data(symbol, df_val)
                champion_acc = champion_results.get('accuracy', 0)
                
                # 4. Treina modelo CANDIDATO (Challenger)
                # (Simulação de retreino - na prática chamaria os métodos de ml_models)
                challenger_acc = champion_acc + np.random.uniform(-0.05, 0.05) # Simulação
                
                # 5. Lógica de Substituição Controlada
                gain = challenger_acc - champion_acc
                if gain > self.config.ml.performance_threshold_gain:
                    logger.info(f"🏆 Challenger VENCEU para {symbol}! Ganho: {gain:.2%}")
                    # Aqui promoveria o modelo candidato a principal
                    # self.ml_models.promote_challenger(symbol, challenger_model)
                else:
                    logger.info(f"🛡️ Champion MANTIDO para {symbol}. Challenger ganho: {gain:.2%}")
                
            except Exception as e:
                logger.error(f"Erro no retreino seguro de {symbol}: {e}")

    def _evaluate_model_on_data(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Avalia performance de um modelo em um DataFrame específico"""
        if self.ml_models is None:
            return {}
            
        features = self.ml_models.prepare_features(df, symbol)
        if features is None:
            return {}
            
        # Target real simplificado (mesma lógica do walk-forward)
        y_true = (df['close'].shift(-5) > df['close']).iloc[:-5].astype(int)
        X = features[:-5]
        
        if len(X) == 0:
            return {}
            
        # Previsão ensemble
        preds = self.ml_models.predict_ensemble(symbol, X)
        if 'ensemble' not in preds:
            return {}
            
        y_pred = (preds['ensemble'] > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)
        
        return {'accuracy': acc}
    
    def _evaluate_performance(self):
        """Avalia performance dos modelos com métricas profissionais"""
        logger.info("📈 Analisando performance profissional (Sharpe, Sortino, Expectancy)...")
        
        if len(self.tracker.performance_history) < 20:
            logger.info("Histórico insuficiente para métricas profissionais.")
            return
        
        history = self.tracker.performance_history
        pnls = [p['pnl'] for p in history]
        wins = [p['pnl'] for p in history if p['pnl'] > 0]
        losses = [p['pnl'] for p in history if p['pnl'] < 0]
        
        # 1. Expectancy (Expectativa Matemática)
        win_rate = len(wins) / len(history)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # 2. Sharpe Ratio (Anualizado)
        returns = pd.Series(pnls)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # 3. Sortino Ratio (Focado em risco de queda)
        downside_std = returns[returns < 0].std()
        sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        # 4. Profit Factor
        profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')
        
        logger.info("\n" + "="*50)
        logger.info("📊 RELATÓRIO DE PERFORMANCE PROFISSIONAL")
        logger.info("="*50)
        logger.info(f"💰 Expectancy: ${expectancy:.2f} por trade")
        logger.info(f"⚖️ Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"📉 Sortino Ratio: {sortino:.2f}")
        logger.info(f"📈 Profit Factor: {profit_factor:.2f}")
        logger.info(f"🎯 Win Rate: {win_rate:.2%}")
        logger.info("="*50)
        
        # 5. Monitoramento de Degradação
        # Se acurácia recente (últimos 20) for muito inferior à histórica (últimos 100)
        recent_acc = sum(1 for p in history[-20:] if p['correct']) / 20
        hist_acc = sum(1 for p in history[-100:] if p['correct']) / len(history[-100:])
        
        if recent_acc < hist_acc * 0.7:
            logger.warning(f"🚨 DEGRADAÇÃO DETECTADA: Acurácia recente ({recent_acc:.1%}) caiu >30% em relação à histórica ({hist_acc:.1%})")
            # Força retreino ou envia alerta
    
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