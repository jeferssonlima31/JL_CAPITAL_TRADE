# 🤖 JL CAPITAL TRADE - Sistema Agressivo de Alta Performance

## 📈 Trading Forex com Machine Learning e Aprendizado Contínuo

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)]()
[![License](https://img.shields.io/badge/license-Proprietary-red)]()

## 🎯 Sobre o Projeto

JL Capital Trade é um sistema profissional de trading automatizado focado em **EUR/USD**, operando com um perfil **Agressivo** e utilizando Machine Learning de alta performance. O sistema foi otimizado para as sessões de Londres e EUA, garantindo máxima liquidez e precisão.

### ✨ Características Principais

- 🧠 **Inteligência Adaptativa**: Ensemble de **XGBoost + MLP** com **Online Learning (SGD)** que aprende com cada trade individualmente.
- 🛡️ **Proteção Robusta**: Sistema de **Heartbeat (1s)**, verificações de **Slippage/Spread** de última milha e **Circuit Breakers** automáticos.
- 📰 **Filtro de Notícias**: Bloqueio automático 30 min antes e 60 min após eventos de alto impacto (USD/EUR).
- 📊 **Análise MTF & Regime**: Filtros de tendência em múltiplos timeframes (H1/H4/D1) e detecção de regime (Tendência vs Range).
- ⚖️ **Gestão Dinâmica**: Risco adaptativo baseado na confiança do modelo e stops dinâmicos calculados pelo **ATR**.
- 🧪 **Validação Rigorosa**: Testado via **Walk-Forward Validation** e **Monte Carlo Simulation** (Probabilidade de Ruína: 0%).

## 🚀 Como Executar

```bash
# Instale as dependências
pip install -r requirements.txt

# Configure suas credenciais no .env
# MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

# Teste a conexão com o MetaTrader 5
python -m jl_capital_trade.cli --test-connection

# Inicie o bot em modo de teste (recomendado para início)
python -m jl_capital_trade.cli --mode test --pair EUR_USD
```

## 🏗️ Arquitetura

O sistema é dividido em módulos especializados:
- `jl_capital_trade.ml_models`: Gerenciamento de modelos e Ensemble.
- `jl_capital_trade.continuous_learning`: Tracker de performance e loop de feedback.
- `jl_capital_trade.trading_bot`: Orquestrador de análise e execução.
- `jl_capital_trade.mt5_connector`: Ponte de comunicação com MetaTrader 5.

---
© 2026 JL Capital Trade. Todos os direitos reservados.
