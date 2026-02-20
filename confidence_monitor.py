#!/usr/bin/env python3
"""
SISTEMA DE MONITORAMENTO DE CONFIANÇA EM TEMPO REAL
JL CAPITAL TRADE
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from integration_system import IntegrationSystem
import logging
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
import os
from dotenv import load_dotenv

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('confidence_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class ConfidenceMonitor:
    def __init__(self):
        self.system = IntegrationSystem()
        self.connected = False
        self.confidence_history = []
        self.performance_metrics = []
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        
        # Limiares de confiança
        self.high_confidence_threshold = 0.7
        self.medium_confidence_threshold = 0.6
        self.low_confidence_threshold = 0.5
        
        logger.info("📊 Sistema de Monitoramento de Confiança Inicializado")
    
    def connect_to_mt5(self):
        """Conecta ao MT5"""
        try:
            if not mt5.initialize():
                return False
            
            authorized = mt5.login(
                login=int(os.getenv('MT5_LOGIN', 3263303)),
                password=os.getenv('MT5_PASSWORD', '!rH5UiSb'),
                server=os.getenv('MT5_SERVER', 'Just2Trade-MT5')
            )
            
            if authorized:
                self.connected = True
                return True
            
        except Exception as e:
            logger.error(f"Erro na conexão: {e}")
        
        return False
    
    def load_ml_system(self):
        """Carrega sistema ML"""
        return self.system.load_best_model()
    
    def analyze_confidence_distribution(self):
        """Analisa distribuição de confiança dos sinais"""
        if not self.connect_to_mt5():
            return
        
        try:
            confidences = []
            signals = []
            
            for symbol in self.symbols:
                # Obtém dados
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
                if rates is None:
                    continue
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Calcula features
                features = self._calculate_features(df)
                if features is None:
                    continue
                
                # Faz previsão
                prediction = self.system.predict(features)
                if prediction:
                    confidences.append(prediction['confidence'][0])
                    signals.append(prediction['prediction'][0])
            
            # Analisa distribuição
            if confidences:
                self._analyze_confidence_stats(confidences, signals)
                
        finally:
            mt5.shutdown()
    
    def _calculate_features(self, df):
        """Calcula features para análise"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1].copy()
        features = {}
        
        # Features básicas
        features['open'] = latest['open']
        features['high'] = latest['high']
        features['low'] = latest['low']
        features['close'] = latest['close']
        features['tick_volume'] = latest['tick_volume']
        features['spread'] = latest['spread']
        features['real_volume'] = latest.get('real_volume', 0)
        
        # Returns
        if len(df) > 1:
            prev_close = df.iloc[-2]['close']
            features['returns'] = (latest['close'] / prev_close) - 1
            features['log_returns'] = np.log(latest['close'] / prev_close)
        else:
            features['returns'] = 0
            features['log_returns'] = 0
        
        # Médias móveis
        features['ma5'] = df['close'].tail(5).mean()
        features['ma20'] = df['close'].tail(20).mean()
        features['ma50'] = df['close'].tail(50).mean()
        features['ma200'] = df['close'].tail(200).mean() if len(df) >= 200 else features['ma50']
        
        # Bollinger Bands
        features['bb_middle'] = features['ma20']
        features['bb_std'] = df['close'].tail(20).std()
        features['bb_upper'] = features['bb_middle'] + 2 * features['bb_std']
        features['bb_lower'] = features['bb_middle'] - 2 * features['bb_std']
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        
        # RSI
        gains = df['close'].diff().clip(lower=0).tail(14).mean()
        losses = (-df['close'].diff().clip(upper=0)).tail(14).mean()
        features['rsi'] = 100 - (100 / (1 + gains/losses)) if losses != 0 else 50
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean().iloc[-1]
        ema26 = df['close'].ewm(span=26).mean().iloc[-1]
        features['macd'] = ema12 - ema26
        features['macd_signal'] = pd.Series([features['macd']]).ewm(span=9).mean().iloc[-1]
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Outros indicadores
        features['volatility'] = df['close'].tail(20).std() / df['close'].tail(20).mean()
        features['atr'] = (df['high'].tail(14) - df['low'].tail(14)).mean()
        features['momentum'] = latest['close'] - df['close'].iloc[-4]
        features['roc'] = (latest['close'] / df['close'].iloc[-10] - 1) * 100
        features['price_vs_ma'] = (latest['close'] / features['ma20'] - 1) * 100
        features['high_low_ratio'] = latest['high'] / latest['low']
        
        return pd.DataFrame([features])
    
    def _analyze_confidence_stats(self, confidences, signals):
        """Analisa estatísticas de confiança"""
        confidences = np.array(confidences)
        signals = np.array(signals)
        
        # Estatísticas básicas
        mean_conf = np.mean(confidences)
        median_conf = np.median(confidences)
        std_conf = np.std(confidences)
        
        # Distribuição por nível de confiança
        high_conf = np.sum(confidences >= self.high_confidence_threshold)
        medium_conf = np.sum((confidences >= self.medium_confidence_threshold) & 
                            (confidences < self.high_confidence_threshold))
        low_conf = np.sum(confidences < self.medium_confidence_threshold)
        
        # Por tipo de sinal
        buy_signals = signals == 1
        sell_signals = signals == 0
        
        buy_conf = np.mean(confidences[buy_signals]) if np.any(buy_signals) else 0
        sell_conf = np.mean(confidences[sell_signals]) if np.any(sell_signals) else 0
        
        logger.info("\n" + "="*60)
        logger.info("📊 ANÁLISE DE CONFIANÇA DOS SINAIS")
        logger.info("="*60)
        logger.info(f"📈 Média: {mean_conf:.3f} | Mediana: {median_conf:.3f} | Std: {std_conf:.3f}")
        logger.info(f"🎯 Alta confiança (>70%): {high_conf} sinais")
        logger.info(f"📊 Média confiança (60-70%): {medium_conf} sinais")
        logger.info(f"⚠️ Baixa confiança (<60%): {low_conf} sinais")
        logger.info(f"🟢 Compra - Confiança média: {buy_conf:.3f}")
        logger.info(f"🔴 Venda - Confiança média: {sell_conf:.3f}")
        
        # Adiciona ao histórico
        self.confidence_history.append({
            'timestamp': datetime.now(),
            'mean_confidence': mean_conf,
            'high_confidence_count': high_conf,
            'signals_count': len(confidences)
        })
    
    def monitor_real_time_confidence(self, duration_minutes=30):
        """Monitora confiança em tempo real"""
        logger.info(f"\n🔍 Iniciando monitoramento em tempo real ({duration_minutes} minutos)")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        cycle_count = 0
        
        while datetime.now() < end_time:
            cycle_count += 1
            
            logger.info(f"\n🔄 Ciclo {cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
            
            # Analisa confiança atual
            self.analyze_confidence_distribution()
            
            # Espera 2 minutos entre ciclos
            time.sleep(120)
        
        logger.info("✅ Monitoramento concluído")
        self.generate_confidence_report()
    
    def generate_confidence_report(self):
        """Gera relatório completo de confiança"""
        if not self.confidence_history:
            logger.warning("⚠️ Nenhum dado de confiança coletado")
            return
        
        # Cria DataFrame com histórico
        df = pd.DataFrame(self.confidence_history)
        
        logger.info("\n" + "="*60)
        logger.info("📈 RELATÓRIO FINAL DE CONFIANÇA")
        logger.info("="*60)
        logger.info(f"📊 Total de ciclos: {len(df)}")
        logger.info(f"🎯 Confiança média geral: {df['mean_confidence'].mean():.3f}")
        logger.info(f"📈 Máxima confiança: {df['mean_confidence'].max():.3f}")
        logger.info(f"📉 Mínima confiança: {df['mean_confidence'].min():.3f}")
        logger.info(f"🔢 Total de sinais analisados: {df['signals_count'].sum()}")
        logger.info(f"⭐ Sinais de alta confiança: {df['high_confidence_count'].sum()}")
        
        # Recomendações
        mean_conf = df['mean_confidence'].mean()
        if mean_conf < 0.55:
            logger.info("\n❌ RECOMENDAÇÃO: Confiança muito baixa")
            logger.info("   🔧 Ações: Retreinar modelos e ajustar parâmetros")
        elif mean_conf < 0.65:
            logger.info("\n⚠️ RECOMENDAÇÃO: Confiança moderada")
            logger.info("   🔧 Ações: Coletar mais dados e otimizar ensemble")
        else:
            logger.info("\n✅ RECOMENDAÇÃO: Confiança adequada")
            logger.info("   🚀 Pronto para trading automático")
    
    def plot_confidence_trend(self):
        """Gera gráfico da tendência de confiança"""
        if not self.confidence_history:
            return
        
        df = pd.DataFrame(self.confidence_history)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(df['timestamp'], df['mean_confidence'], 'b-', marker='o')
        plt.title('Tendência da Confiança Média')
        plt.xlabel('Tempo')
        plt.ylabel('Confiança Média')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.bar(['Alta', 'Média', 'Baixa'], 
                [df['high_confidence_count'].sum(), 
                 len(df) * 4 - df['high_confidence_count'].sum(),
                 0])  # Simplificado
        plt.title('Distribuição de Confiança')
        plt.ylabel('Quantidade')
        
        plt.tight_layout()
        plt.savefig('confidence_trend.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Gráfico de tendência salvo: confidence_trend.png")

def main():
    """Função principal"""
    monitor = ConfidenceMonitor()
    
    # Carrega sistema ML
    if not monitor.load_ml_system():
        logger.error("❌ Falha ao carregar sistema ML")
        return
    
    # Monitora confiança em tempo real
    monitor.monitor_real_time_confidence(duration_minutes=15)
    
    # Gera relatório
    monitor.generate_confidence_report()
    monitor.plot_confidence_trend()

if __name__ == "__main__":
    main()