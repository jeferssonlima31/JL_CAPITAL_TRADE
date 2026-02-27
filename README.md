# 🤖 JL CAPITAL TRADE - Sistema Agressivo de Alta Performance

## 📈 Trading Forex com Machine Learning e Aprendizado Contínuo

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)]()
[![License](https://img.shields.io/badge/license-Proprietary-red)]()

## 🎯 Sobre o Projeto

JL Capital Trade é um sistema profissional de trading automatizado focado em **EUR/USD**, operando com um perfil **Agressivo** e utilizando Machine Learning de alta performance. O sistema foi otimizado para as sessões de Londres e EUA, garantindo máxima liquidez e precisão.

### ✨ Características Principais

- 🧠 **ML de Alta Performance**: Ensemble de XGBoost e MLP (Neural Network) com 43 features técnicas avançadas.
- � **Perfil Agressivo**: Configurado para 5% de risco por trade com Relação Risco:Retorno de 1:4.
- � **Filtro de Sessões**: Operação focada em Londres (08h-17h) e EUA (13h-22h) - Horário de Portugal.
- 🔄 **Aprendizado Contínuo**: Loop de feedback real que ajusta pesos dos modelos e retreina baseado em performance.
- 🔒 **Segurança**: Criptografia de credenciais, auditoria de ações e gerenciamento de risco rigoroso.
- 📊 **Interface CLI**: Controle total via linha de comando para monitoramento e análise.

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
