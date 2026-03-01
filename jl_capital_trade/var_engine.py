# =============================================================================
# JL CAPITAL TRADE - VaR ENGINE (VALUE AT RISK)
# =============================================================================

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class VaRResult:
    """Resultado Categórico da Análise Dinâmica VaR"""
    value_at_risk_currency: float
    value_at_risk_percent: float
    is_safe: bool
    confidence_level: float
    horizon_hours: int
    worst_case_scenario: float

class VaREngine:
    """
    Motor Institucional de Value at Risk (VaR)
    Utiliza simulações de Monte Carlo (Geometia Browniana) para prever o pior cenário
    matemático de um trade antes ou durante a sua execução.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Parâmetros padrão do algoritmo Monte Carlo
        self.simulations = 10000
        self.confidence_level = 0.99  # 99% de certeza estatística
        
        # Limites máximos de resiliência (Exness / Trade)
        # Se 99% dos cenários derem perda maior que isso, abortamos a execução.
        self.max_var_percent_per_trade = 0.05  # Máximo 5% de chance de ruína na conta
        
        logger.info("🛡️ Motor VaR (Monte Carlo) Inicializado")

    def _simulate_price_paths(self, current_price: float, volatility: float, 
                              drift: float, periods: int) -> np.ndarray:
        """
        Gera milhares de caminhos futuros baseados em Movimento Browniano Geométrico (GBM).
        """
        dt = 1 / 252 # Tempo estendido para cálculo diário
        
        # Cria matriz de choques aleatórios normais (simulations x periods)
        random_shocks = np.random.normal(0, 1, size=(self.simulations, periods))
        
        # Calcula fator de crescimento dos caminhos
        drift_factor = (drift - 0.5 * volatility ** 2) * dt
        volatility_factor = volatility * np.sqrt(dt) * random_shocks
        
        # Acumula retornos para gerar as linhas de preços futuros
        daily_returns = np.exp(drift_factor + volatility_factor)
        price_paths = np.zeros_like(daily_returns)
        price_paths[:, 0] = current_price
        
        for t in range(1, periods):
            price_paths[:, t] = price_paths[:, t-1] * daily_returns[:, t]
            
        return price_paths

    def calculate_var(self, symbol: str, current_price: float, position_volume: float, 
                      balance: float, historical_returns: pd.Series, horizon_hours: int = 24) -> Optional[VaRResult]:
        """
        Calcula o Value at Risk usando Histórico + Monte Carlo.
        
        Args:
            symbol: Ativo ('EUR_USD', etc)
            current_price: Preço de cotação agora
            position_volume: Lotes do MT5 (ex: 0.10)
            balance: Saldo atual da conta
            historical_returns: Series do pandas com os retornos logarítmicos recentes do ativo
            horizon_hours: Daqui quantas horas queremos projetar o risco?
        """
        if balance <= 0 or historical_returns is None or len(historical_returns) < 50:
            logger.warning(f"⚠️ VaR Motor sem dados suficientes para {symbol}.")
            return None
            
        try:
            # 1. Extração Estatística do Comportamento Recente (Volatility e Drift/Mean)
            volatility = historical_returns.std() * np.sqrt(252) # Anualizada
            drift = historical_returns.mean() * 252 # Drift anualizado
            
            # Ajuste de segurança em mercados parados
            if volatility < 0.0001:
                volatility = 0.05 # Puxa Volatilidade estática basal de 5%
                
            # 2. Simulação Monte Carlo p/ Preço Futuro
            price_paths = self._simulate_price_paths(current_price, volatility, drift, horizon_hours)
            
            # Pega todos os 10.000 cenários finais da última coluna (A imagem do futuro após N horas)
            final_prices = price_paths[:, -1]
            
            # 3. Cálculo de Risco P/L
            # Valor do Ponto (Dólar vs EUR) padrão 100k
            point_value = 100000 if "EUR_USD" in symbol else 100
            
            # Calcula PnL de todos os 10.000 cenários se tivermos Comprado "BUY" (Pior caso possível de queda)
            pnls = (final_prices - current_price) * position_volume * point_value
            
            # 4. Encontra o Value at Risk no Pior Percentil escolhido (1% de cauda)
            worst_case_index = int((1 - self.confidence_level) * self.simulations)
            sorted_pnls = np.sort(pnls)
            var_dollar_loss = abs(sorted_pnls[worst_case_index]) # VaR é sempre positivo no relatório
            worst_case_loss = abs(sorted_pnls[0])
            
            var_percent = var_dollar_loss / balance
            
            # 5. Veredito do Escudo de Risco
            is_safe = var_percent <= self.max_var_percent_per_trade
            
            # Feedback e Métricas de Emergência
            if not is_safe:
                logger.error(f"🔴 VaR REJEITADO: {symbol} - Alocação {position_volume} Lotes arrisca ${var_dollar_loss:.2f} ({(var_percent*100):.1f}% da conta) em 99% das vezes.")
            else:
                logger.debug(f"🟢 VaR SEGURO: {symbol} tem risco simulado máximo de ${var_dollar_loss:.2f} ({(var_percent*100):.2f}%).")
                
            return VaRResult(
                value_at_risk_currency=var_dollar_loss,
                value_at_risk_percent=var_percent,
                is_safe=is_safe,
                confidence_level=self.confidence_level,
                horizon_hours=horizon_hours,
                worst_case_scenario=worst_case_loss
            )
            
        except Exception as e:
            logger.error(f"Erro Crítico no VaREngine para {symbol}: {e}")
            return None
