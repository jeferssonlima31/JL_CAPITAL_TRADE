# 🏗️ Arquitetura do Sistema - JL CAPITAL TRADE

O sistema foi desenvolvido sob uma arquitetura modular, focada em escalabilidade, segurança e alta performance para trading algorítmico.

## 📁 Estrutura de Diretórios

```
JL_CAPITAL_TRADE/
├── jl_capital_trade/          # Núcleo do sistema (Package)
│   ├── cli.py                 # Interface de linha de comando (Entrypoint)
│   ├── config.py              # Gerenciador de configurações e variáveis de ambiente
│   ├── ml_models.py           # Gestão de modelos (XGBoost, MLP/LSTM) e Ensemble
│   ├── continuous_learning.py # Loop de feedback e tracker de performance
│   ├── trading_bot.py         # Orquestrador de análise e execução de ordens
│   ├── mt5_connector.py       # Integração direta com MetaTrader 5 API
│   ├── data_collector.py      # Coleta e cálculo de indicadores técnicos
│   ├── risk_manager.py        # Gerenciamento de risco e cálculo de lotes
│   └── security.py            # Criptografia, JWT e Auditoria
├── trained_models/            # Modelos serializados (.joblib, .pkl, .model)
├── docs/                      # Documentação técnica
├── logs/                      # Logs de execução e auditoria
└── .env                       # Credenciais e parâmetros sensíveis
```

## 🧠 Fluxo de Machine Learning (Modo Agressivo)

O sistema opera com um pipeline de dados robusto:
1. **Coleta**: O `data_collector` busca 500 barras de histórico via `mt5_connector`.
2. **Features**: O `aggressive_feature_compatibility` calcula 43 indicadores técnicos (RSI, MACD, BB, Volatilidade, Sessões, etc).
3. **Previsão**: O `ml_models` utiliza um Ensemble ponderado de **XGBoost** e **MLP Classifier**.
4. **Feedback**: Após cada trade, o `continuous_learning` recebe o resultado real e ajusta os pesos do Ensemble para as próximas operações.

## 🕒 Filtro de Sessões e Liquidez

O bot monitora o horário de Portugal para operar apenas quando a volatilidade e o volume são ideais para o EURUSD:
- **Londres (08h-17h)**
- **EUA (13h-22h)**

## 🛡️ Gerenciamento de Risco

Configurado para perfil agressivo:
- **Risco por Trade**: 5% do capital.
- **Stop Loss**: 30 pips.
- **Take Profit**: 120 pips (Relação 1:4).
- **Limite Diário**: 10% de perda máxima (Daily Loss Limit).

---
© 2026 JL Capital Trade.
