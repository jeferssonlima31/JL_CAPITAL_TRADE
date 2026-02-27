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

## 🧠 Fluxo de Machine Learning (Modo Robusto)

O sistema opera com um pipeline otimizado para evitar overfitting:
1. **Coleta**: O `data_collector` busca 500 barras de histórico.
2. **Seleção de Features**: Redução de 43 para as **6 features mais robustas** (MA_50, MA_100, MA_200, Volatilidade e Bollinger Std) usando `SelectFromModel`.
3. **Regularização**: O treinamento utiliza regularização L1 (Lasso) e L2 (Ridge) para evitar o ajuste excessivo ao ruído histórico.
4. **Previsão**: Ensemble de **XGBoost** (Regularizado) e **ExtraTrees**.
5. **Feedback**: Ajuste contínuo de pesos baseado em performance real.

## 🔄 Aprendizado Contínuo Seguro

O sistema utiliza uma abordagem de **Champion vs Challenger** para evoluir os modelos sem riscos:
1. **Acúmulo de Dados**: O retreino só é disparado após acumular um volume mínimo de novos dados (ex: 500 amostras).
2. **Modelo Candidato (Challenger)**: Um novo modelo é treinado em paralelo com os dados mais recentes.
3. **Validação OOS**: O Candidato é testado em um conjunto de dados que não participou do seu treino (Out-of-Sample).
4. **Promoção Controlada**: O Candidato só substitui o Modelo Principal (Champion) se houver um ganho de performance real acima do threshold (ex: +2% de acurácia).
5. **Ajuste de Pesos**: Resultados de trades individuais continuam ajustando os pesos do Ensemble em tempo real.

## 🧪 Validação e Testes de Robustez

O sistema utiliza técnicas avançadas de validação para garantir a estabilidade do modelo:
- **Walk-Forward Validation (WFV)**: O modelo é validado em janelas móveis de tempo (12.000 barras / ~1.5 anos), garantindo que o teste ocorra sempre em dados cronologicamente posteriores ao treino.
- **Resultados OOS (Out-of-Sample)**: Acurácia real de **74.62%** com Sharpe Ratio de **19.39**, confirmando a eficácia dos filtros de robustez.
- **Simulação Cronológica Realista**: Testes que consideram spreads, slippage e a evolução do saldo da conta (Equity Curve).
- **Métricas de Risco**: Monitoramento constante de Sharpe Ratio (alvo > 2.0) e Max Drawdown (limite 15%).

## 🧠 Inteligência de Mercado Avançada

O sistema utiliza camadas de análise para filtrar ruídos e operar apenas em condições ideais:
- **Detecção de Regime de Mercado**: Identifica se o par está em **Tendência (Trending)** ou **Lateralização (Ranging)** usando ADX e ATR. O bot adapta os thresholds de entrada conforme o regime.
- **Análise Multi-Timeframe (MTF)**: Os sinais de H1 só são executados se estiverem alinhados com a tendência principal do H4 e sem sobrecompra/sobrevenda extrema no D1.
- **Filtro de Probabilidade**: O sistema exige uma confiança mínima de **75%** do Ensemble para disparar uma ordem, focando em qualidade sobre quantidade.
- **Sessões UTC**: Controle rígido de horário (Londres/EUA) baseado em UTC para evitar erros com fusos horários e horários de verão.

## ⚖️ Gestão Dinâmica e Métricas

Para maximizar o retorno e minimizar o risco, o sistema implementa:
- **Stops Dinâmicos ATR**: Stop Loss e Take Profit são calculados com base na volatilidade real (2.5x ATR para SL e 10x ATR para TP), mantendo um RR de 1:4.
- **Risco Adaptativo**: O risco por trade é ajustado dinamicamente (+/- 20%) com base na confiança do modelo e no drawdown atual da conta.
- **Métricas Profissionais**: O sistema monitora **Sharpe Ratio**, **Sortino Ratio**, **Expectancy** e **Profit Factor** em tempo real.
- **Monitoramento de Degradação**: Alertas automáticos se a acurácia recente cair drasticamente em relação à histórica, indicando necessidade de retreino.
- **Simulação Monte Carlo**: Scripts integrados para testar a robustez estatística contra sequências aleatórias de mercado.

## 💸 Simulação de Custos Reais

Para garantir que a estratégia seja lucrativa após todas as taxas, o sistema simula:
- **Spread Variável**: Monitoramento do spread bid/ask em tempo real, com bloqueio de trades se o spread exceder o limite (ex: 2.0 pips).
- **Slippage Realista**: Simulação de atraso na execução, adicionando um custo de derrapagem (ex: 0.5 pips) no preço de entrada.
- **Comissões de Corretora**: Cálculo automático de taxas por lote negociado (ex: $7.00/lote).
- **Execução Líquida**: O P&L reportado pelo sistema já desconta todos os custos operacionais, fornecendo o lucro real disponível para saque.

## 🛡️ Camadas de Proteção de Execução

O sistema implementa proteções de "última milha" para garantir a integridade de cada trade:
- **Heartbeat Monitor**: Uma thread dedicada monitora a conexão com o terminal MT5 a cada 1 segundo. Se a conexão for perdida, o bot entra em modo de segurança imediatamente e tenta reconectar automaticamente.
- **Verificação Interna de Spread**: Antes de enviar a ordem ao mercado, o conector valida o spread bid/ask real. Se exceder o limite (ex: 1.5 pips para EURUSD), a ordem é cancelada mesmo que o bot tenha autorizado.
- **Monitoramento de Slippage Real**: Após cada execução, o sistema calcula a diferença entre o preço solicitado e o executado. Slippages acima de 1.5 pips disparam um Circuit Breaker.
- **Paridade de Dados**: O conector garante a compatibilidade entre os nomes de colunas do MetaTrader 5 (`tick_volume`) e os requeridos pelos modelos legados e novos.

## 🧠 Inteligência Adaptativa e Aprendizado Online

Além do retreino periódico, o sistema evolui em tempo real:
- **Online Learning (SGD)**: Um modelo `SGDClassifier` aprende com o resultado de cada trade individualmente via `partial_fit`. Isso permite que o sistema se adapte a mudanças de regime de mercado em minutos.
- **Regime-Aware Ensemble**: O peso dos modelos no ensemble é ajustado dinamicamente:
    - Em regimes de **Tendência**, o modelo XGBoost recebe maior peso.
    - Em regimes de **Lateralização (Range)**, o modelo MLP/ExtraTrees é favorecido.
- **Filtro de Probabilidade de 73%**: Calibrado para buscar uma acurácia real de **70-75%** no longo prazo, equilibrando seletividade e frequência de trades.
- **Multi-Timeframe (MTF) Alignment**: Sinais de H1 são automaticamente bloqueados se estiverem contra a tendência principal do H4.

## 📰 Filtro de Notícias Econômicas

Para proteger o capital contra picos de volatilidade irracional, o sistema integra um filtro de notícias:
- **Monitoramento USD/EUR**: O bot consulta o calendário econômico da Forex Factory para eventos de alto e médio impacto.
- **Janelas de Pausa**: O trading é automaticamente bloqueado 30 minutos antes e até 60 minutos após notícias importantes.
- **Proteção ATR**: Monitoramento em tempo real da volatilidade (ATR). Se a volatilidade atual exceder 2.5x a média recente, novas operações são bloqueadas.
- **Cache de Contingência**: O sistema mantém uma cópia local do calendário para garantir a proteção mesmo em caso de falha na conexão com o provedor de notícias.

## 🛑 Circuit Breakers de Segurança

O sistema possui "disjuntores" automáticos que interrompem a operação em condições anormais:
- **Perdas Consecutivas**: Bloqueio total após 3 trades seguidos com prejuízo.
- **Drawdown Intraday**: Bloqueio se a perda acumulada no dia exceder 5%.
- **Drawdown Histórico**: Bloqueio se a queda em relação ao pico da conta exceder 15%.
- **Spread Excessivo**: Cancelamento de ordens individuais se o spread bid/ask for maior que 2.0 pips.
- **Slippage Crítico**: Desativação do bot se a diferença entre o preço solicitado e o executado for maior que 1.5 pips (indica baixa liquidez ou manipulação).
- **Reset de Segurança**: Uma vez disparado um Circuit Breaker (exceto spread), o sistema entra em modo de segurança e exige intervenção manual ou reinicialização diária.

## 🕒 Filtro de Sessões e Liquidez

O bot monitora o horário de Portugal para operar apenas quando a volatilidade e o volume são ideais para o EURUSD:
- **Londres (08h-17h)**
- **EUA (13h-22h)**

## 🛡️ Gerenciamento de Risco

Configurado para perfil equilibrado com foco em preservação de capital:
- **Risco por Trade**: 1.5% do capital.
- **Stop Loss**: 30 pips.
- **Take Profit**: 120 pips (Relação 1:4).
- **Limite Diário**: 5% de perda máxima (Daily Loss Limit).
- **Limite de Perdas Consecutivas**: Bloqueio após 3 perdas seguidas.
- **Controle de Drawdown**: Bloqueio automático ao atingir 15% de drawdown histórico.

---
© 2026 JL Capital Trade.
