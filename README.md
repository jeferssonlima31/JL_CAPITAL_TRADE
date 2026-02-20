# 🤖 JL CAPITAL TRADE

## Sistema Profissional de Trading Forex com Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-Proprietary-red)]()
[![OpenClaw](https://img.shields.io/badge/OpenClaw-Integrated-green)]()

## 🎯 Sobre o Projeto

JL Capital Trade é um sistema profissional de trading automatizado para Forex, focado em **EUR/USD** e **XAU/USD**, utilizando Machine Learning avançado com aprendizado contínuo e integração com **OpenClaw**.

### ✨ Características

- 🧠 **Machine Learning**: XGBoost, LSTM, Ensemble com aprendizado contínuo
- 🔒 **Segurança**: Criptografia, JWT, auditoria, rate limiting
- 💾 **Cache**: Redis + memória, 90% menos chamadas à API
- 📊 **Interface**: CLI + API REST
- 🤖 **OpenClaw**: Integração completa via skills
- 📈 **Backtesting**: Simulação histórica
- 🔄 **Aprendizado Contínuo**: Modelos evoluem com feedback

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/seu-usuario/jl-capital-trade.git
cd jl-capital-trade

# Configure
cp .env.example .env
# Edite .env com suas credenciais da Exness

# Instale
pip install -r requirements.txt

# Teste conexão
python -m jl_capital_trade.cli --test-connection

# Execute
python -m jl_capital_trade.cli --mode test