#!/usr/bin/env python3
"""
MONTE CARLO SIMULATION - JL CAPITAL TRADE
Testa a robustez estatística da estratégia simulando milhares de sequências aleatórias.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_monte_carlo(n_simulations=1000, n_trades=100, win_rate=0.67, 
                    avg_win=4.0, avg_loss=1.0, initial_balance=100000):
    """
    Simula sequências de trades para entender o risco de ruína e variação de capital.
    avg_win/avg_loss representa o Risk/Reward ratio (ex: 1:4).
    """
    logger.info(f"🚀 Iniciando Monte Carlo: {n_simulations} simulações de {n_trades} trades...")
    
    all_results = []
    ruin_count = 0
    ruin_threshold = initial_balance * 0.5 # Perder 50% é considerado ruína técnica
    
    for _ in range(n_simulations):
        balance = initial_balance
        equity_curve = [balance]
        
        # Simula sequência de trades
        outcomes = np.random.choice([1, -1], size=n_trades, p=[win_rate, 1-win_rate])
        
        for outcome in outcomes:
            risk_per_trade = balance * 0.015 # 1.5% de risco fixo
            
            if outcome == 1:
                balance += risk_per_trade * avg_win
            else:
                balance -= risk_per_trade * avg_loss
                
            equity_curve.append(balance)
            
            if balance < ruin_threshold:
                ruin_count += 1
                break
                
        all_results.append(equity_curve[-1])
        
    # Estatísticas
    final_balances = np.array(all_results)
    prob_ruin = ruin_count / n_simulations
    expected_return = (np.mean(final_balances) / initial_balance - 1) * 100
    median_return = (np.median(final_balances) / initial_balance - 1) * 100
    
    logger.info("\n" + "="*50)
    logger.info("📊 RESULTADOS DA SIMULAÇÃO MONTE CARLO")
    logger.info("="*50)
    logger.info(f"💰 Retorno Médio Esperado: {expected_return:.2f}%")
    logger.info(f"🎯 Retorno Mediano: {median_return:.2f}%")
    logger.info(f"💀 Probabilidade de Ruína (>50% Drawdown): {prob_ruin:.2%}")
    logger.info(f"🏆 Melhor Cenário: ${(np.max(final_balances)):,.2f}")
    logger.info(f"📉 Pior Cenário: ${(np.min(final_balances)):,.2f}")
    logger.info("="*50)
    
    return final_balances

if __name__ == "__main__":
    # Baseado nos resultados reais do nosso modelo robusto
    run_monte_carlo(win_rate=0.67, avg_win=4.0, avg_loss=1.0)
