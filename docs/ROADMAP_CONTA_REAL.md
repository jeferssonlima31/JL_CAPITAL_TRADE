# Roadmap para Conta Real & Futuro Pós-Live (JL Capital Trade)

Este documento dita o protocolo estrito de evolução do projeto desde os testes na Conta Demo até a implantação em grande escala na Conta Real, estabelecendo metas e métricas a serem batidas em cada fase, como num fundo quantitativo sério.

---

## FASE 1: O Campo de Provas (Conta Demo Exness)
**Duração Estimada:** Mínimo de 30 dias corridos ou 100 operações (o que ocorrer primeiro).
**Objetivo:** Validar a resiliência mecânica do código (se ele trava), a latência para a corretora e o alinhamento da Inteligência Artificial em um ambiente simulando dinheiro real.

### Checklist para Avanço (O que precisa acontecer para irmos para a Real):
1. [ ] **Sobrevivência Técnica Geral:** O robô funcionou por 3 semanas sem *Crashes*, *Memory Leaks* ou travamentos no loop principal?
2. [ ] **Métricas da IA:** A proporção de acerto (Win Rate) ficou estatisticamente acima de 55% num risco/retorno de 1:1.5? Ele está executando compras/vendas nos mesmos padrões lógicos que os testes em backtest disseram que ele faria?
3. [ ] **Gerenciamento de Risco à Prova de Fogo:** Ele cortou posições com precisão cirúrgica ao esbarrar no *Circuit Breaker* de Drawdown Estipulado?
4. [ ] **Recuperação de Interrupções:** Forçamos o fechamento do Python enquanto ele tinha uma operação aberta na Exness e religamos. Ele reconheceu a operação flutuando, adotou ela para sua contabilidade e continuou a monitorar sem pirar?
5. [ ] **Reagindo à Tempestade:** O motor VaR abortou posições corretamente em momentos raros de volatilidade macro insana?

---

## FASE 2: Conta Real com "Scouting" (Micro-Lotes)
**Duração Estimada:** Mínimo de 15 a 21 dias (Pode ser uma conta de $1000 com alocação minúscula ou uma subconta `Cent` com lotes de 0.01).
**Objetivo:** Ver se o estresse de slippage (escorregamento de preço da ordem da Corretora enviando pro banco) afeta a métrica que tínhamos na Demo.

### Checklist (Validando o comportamento com Dinheiro Real):
1. [ ] **Auditoria de Slippage:** Os preços que a IA decidiu comprar bateram com o preço final em que o trade foi firmado em 90% das vezes ou The Spread Range devorou nosso lucro? 
2. [ ] **Confiança Psicológica:** O painel log não apresenta nenhum *Warning*. O modelo continuou performando na taxa estatística prevista. Lucro foi gerado sobre capital líquido em um mini ambiente validado!

---

## FASE 3: Full Automation & Scaling (Capital Completo)
**Duração:** Indefinido (O estado da Arte para colher lucros de forma passiva).
**Objetivo:** Alocação total dos fundos originais na Corretora Exness, monitoramento diário do `Log` com intervenção zero.

Nesta fase, você deixa o robô solto enquanto cuida do seu negócio ou dorme. Intervir apenas quando ele enviar uma notificação por e-mail/Telegram acusando problemas severos ("Circuit Breaker Activado").

---

## FASE 4: O "Fim de Jogo" Pós-Conta Real (Expansão e Gestão de Terceiros)
Quando o Robô estiver há 90+ Dias (3 Meses) operando na sua Conta Real rendendo resultados sólidos com rebaixamento (Drawdown) menor do que 10%, ele deixa de ser um experimento e passa ser um **Ativo Financeiro (Produto)**. 

### Degraus Pós-Real:
1. **Implantação de PAMM / MAM (CopyTrading):** A Exness suporta que o seu Master Account permita que outras pessoas do planeta vinculem o dinheiro delas no seu Robô em troca de 30% da lucratividade. O robô opera 1 lote pra você e divide os lotes para os clientes dele em background. Você acabou de fundar um Fundo Quantitativo Automatizado.
2. **Integração de VPS Institucional:** Migração do robô do seu ambiente para um Servidor Privado Virtual rodando Linux em Nuvem (AWS EC2), disponível 24/7/365, protegido de quedas de internet e energia na sua casa.
3. **Paineis de Notificação Telegram/Web:** Adição de um Módulo Frontend (HTML/CSS Dinâmico que já possuímos habilidades para fazer) num painel da Web, onde de qualquer lugar do mundo você abre seu celular e vê um relógio digital com as estatísticas em tempo real, os logs coloridos do robô e botões virtuais para dar **Pausas de Emergência** via internet.

---
**Data de Criação do Roadmap:** 01 de Março de 2026.
