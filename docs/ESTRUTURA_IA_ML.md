# Arquitetura de Inteligência Artificial e Machine Learning
**JL Capital Trade - Sistema Institucional**

O sistema funciona sobre uma arquitetura de **Ensemble Machine Learning** (Aprendizado de Máquina em Conjunto). Em vez de depender de um único "cérebro" de previsão, o robô treina múltiplos algoritmos que votam democraticamente e têm suas confianças calibradas estatisticamente.

As bibliotecas base utilizadas são `scikit-learn`, `xgboost`, `shap` e `numpy`, permitindo processamento lógico ultra-rápido localmente adaptado para CPUs modernas sem exigência estrita de clusters de GPU.

Abaixo está o detalhamento dos 4 pilares do Cérebro Artificial do Projeto:

## 1. O Motor Principal (Séries Temporais Rápidas)
Estes são os algoritmos centrais para captar direções imediatas (Momentum) nos gráficos intraday (Ex: H1).

* **XGBoost (eXtreme Gradient Boosting):** O motor de predição primário. Reconhecido por vencer vastas competições mundiais de dados não-lineares. Ele constrói sequências de árvores de decisão onde cada nova árvore penaliza os erros da árvore anterior, resultando em precisão afiada contra o ruído natural do Forex.
* **Random Forest (Floresta Aleatória):** O contrapeso de segurança. Ele instancia 1.000 árvores de predição avaliando recortes minúsculos de features de mercado isoladamente (Data Bagging). Reduz severamente as "alucinações" que modelos mais agressivos como o XGBoost teriam em mercados instáveis.

## 2. Padrões Históricos Complexos e Geometria de Dados
Adicionados para substituir as pesadas Redes LSTMs (Tensorflow) que geravam incompatibilidades contínuas de ambiente e latência.

* **MLP Classifier (Multi-Layer Perceptron):** Uma Rede Neural clássica baseada no *Scikit-Learn*. Com camadas ocultas densas de neurônios puros, o papel do MLP é correlacionar "padrões mortos" distantes (Ex: *"Identifiquei este setup 800 horas atrás e o mercado subiu"*), complementando as árvores.
* **SVM (Support Vector Machine):** Utiliza matemática espacial de vetores para traçar hiperplanos (Muros 3D) invisíveis separando as características de um trade que "Deu Lucro" e "Deu Prejuízo", focando cirurgicamente no limiar estreito da tomada de decisão ótima.

## 3. Calibrador Probabilístico (Meta-Modelo Sensato)
A função do Calibrador é duvidar da certeza crua emitida pelas IAs.
* **CalibratedClassifierCV (Isotonic Scaling):** Se o modelo primário afirma: *"Estou 85% confiante de que o EURUSD vai subir"*, este meta-algoritmo recusa aceitar cegamente. Ele mapeia o histórico de "quando ele exclamou 85% de certeza, qual foi a taxa real empírica de acerto?". Se no passado a acurácia real nesse cenário for apenas 60%, a saída exibida nos logs e passada ao *Risk Manager* é corrigida para uma severa "Confiança de 60%".

## 4. O Sistema de "Anjos da Guarda" (Risk Machine Learning)
Modelos de aprendizado implementados pura e restritamente para **defesa do capital** contra as falhas estatísticas de longo prazo.

* **SHAP (Shapley Additive exPlanations):** Conceito nativo de *Teoria dos Jogos* provado ser anti-overfitting. Ele abre o XGBoost na fase de Retreino e arranca da matemática as colunas (Features) inúteis ("O Preço de Fechamento de Sexta" ou um Indicador X distorcido). Se a variável não comprovar de fato que causou o lucro no passado, ela é descartada do modelo. O Bot só avalia o Top 15 dados mais purificados.
* **PSI (Population Stability Index):** O *Tracker* de Mutação Macroeconômica (*Data Drift*). Baseado em matemática de Divergência de Kullback-Leibler, ele mede se a "Sexta-Feira louca de hoje" ficou fora do espectro de normalidade de quando o Robô foi treinado há semanas. Se a distância der > 0.2, a IA tranca o botão de COMPRAR e alerta que o modelo quebrou e exige retreinamento.
* **Monte Carlo / VaR Dinâmico:** Um esticador de túnel do tempo em instâncias vivas. A cada *tick* da ação de uma posição ativa, 10.000 cópias prováveis (baseado em Geometria Browniana e Volatilidade instantânea do ativo) são traçadas no futuro. Se >1% dos trajetos indicarem uma quebra catastrófica de 5% da conta, a IA executa o Aborto Emergencial Automático do mercado de imediato.

---
**Data de Compilação:** 01 de Março de 2026.
