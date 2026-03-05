#!/usr/bin/env python
# =============================================================================
# JL CAPITAL TRADE — BOT AO VIVO
# Inicia o bot de trading em modo desenvolvimento (sem trades reais)
# Para trading real: altere ENVIRONMENT=production no .env
# =============================================================================

import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np

# Força UTF-8 para evitar UnicodeEncodeError no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Configura logging com formato claro
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_log.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from jl_capital_trade.config import config
from jl_capital_trade.mt5_connector import MT5Connector
from jl_capital_trade.data_collector import DataCollector
from jl_capital_trade.ml_models import JLMLModels
from jl_capital_trade.risk_manager import RiskManager
from jl_capital_trade.cache_manager import CacheManager


def print_banner():
    print("\n" + "=" * 65)
    print("  🤖  JL CAPITAL TRADE — Trading Bot ao Vivo")
    print("  📅  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("  ⚙️   Ambiente: " + config.environment.value.upper())
    print("=" * 65 + "\n")


def print_account_status(acc: dict):
    print("\n  📊 INFORMAÇÕES DA CONTA:")
    print(f"    Login      : {acc['login']}")
    print(f"    Saldo      : ${acc['balance']:>12,.2f} {acc['currency']}")
    print(f"    Equity     : ${acc['equity']:>12,.2f} {acc['currency']}")
    print(f"    Margem Liv.: ${acc['margin_free']:>12,.2f} {acc['currency']}")
    print(f"    Alavancagem: 1:{acc['leverage']}")
    print()


def analyze_pair(symbol_mt5: str, symbol_key: str,
                 dc: DataCollector, ml: JLMLModels,
                 cache: CacheManager) -> dict | None:
    """Analisa um par e retorna sinal ou None"""

    # Verifica cache
    cache_key = f"signal_{symbol_key}_H1"
    cached = cache.get(cache_key)
    if cached:
        return cached

    # Coleta dados passando o symbol_key para que o connector aplique o sufixo
    df = dc.get_historical_data(symbol_key, 'H1', 500)
    if df is None or len(df) < 100:
        return None

    # Indicadores
    df = dc.calculate_indicators(df, symbol_key)

    # Features
    features = ml.prepare_features(df, symbol_key)
    if features is None:
        return None

    models_available = ml.get_model_list(symbol_key)
    if not models_available:
        return {'symbol': symbol_key, 'action': 'HOLD',
                'reason': 'Sem modelos treinados', 'confidence': 0}

    # Lookback
    lookback = config.ml.eurusd_lookback

    if len(features) < lookback:
        return None

    X = features[-lookback:].reshape(1, lookback, features.shape[1])
    predictions = ml.predict_ensemble(symbol_key, X)

    if not predictions or 'ensemble' not in predictions:
        return None

    ens_val = predictions.get('ensemble', 0.5)
    ens = float(ens_val[0] if isinstance(ens_val, (list, np.ndarray)) else ens_val)
    confidence = abs(ens - 0.5) * 2
    current_price = float(df['close'].iloc[-1])
    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0

    # Usar a decisão unificada do Sistema de Votação (Multi-Agent Consensus)
    consensus = predictions.get('consensus', {})
    action = consensus.get('action', 'HOLD')
    is_unanimous = consensus.get('unanimous', False)
    
    # Confiança baseada nos Votos Reais em vez da probabilidade pura
    buy_votes = consensus.get('buy_votes', 0)
    sell_votes = consensus.get('sell_votes', 0)
    total_voters = consensus.get('total_voters', 1)
    
    if action != 'HOLD':
        # Calcula taxa de confianca real nas votações (ex: 3/3 = 100%, 2/3 = 66%)
        winning_votes = buy_votes if action == 'BUY' else sell_votes
        confidence = float(winning_votes) / float(max(1, total_voters))
        signal_comment = f"CONSENSUS_{action} ({winning_votes}/{total_voters})"
        if is_unanimous:
            signal_comment += " UNANIME"
    else:
        # Se for HOLD (Empate ou Votos fracos, não opera)
        confidence = 0.0
        signal_comment = "HOLD_NO_CONSENSUS"

    signal = {
        'symbol'      : symbol_key,
        'action'      : action,
        'confidence'  : confidence,
        'ensemble'    : ens,
        'price'       : current_price,
        'atr'         : atr,
        'models_used' : list(predictions.keys()),
        'timestamp'   : datetime.now().isoformat(),
        'tactical_c'  : signal_comment
    }

    if action != 'HOLD':
        cache.set(signal, cache_key, ttl=900)  # cache por 15 min

    return signal


def print_signal(signal: dict):
    icon  = '🟢' if signal['action'] == 'BUY' else ('🔴' if signal['action'] == 'SELL' else '⚪')
    conf  = signal.get('confidence', 0)
    price = signal.get('price', 0)
    tac   = signal.get('tactical_c', 'Normal')
    print(f"  {icon} {signal['symbol']:<10} | {signal['action']:<5} | "
          f"Confiança: {conf:.1%} | Preço: {price:.5f} | "
          f"Tático: {tac}")


def main():
    print_banner()

    # ── Conecta MT5 ──────────────────────────────────────────────────────────
    logger.info("🔗 Conectando ao Exness MT5...")
    mt5 = MT5Connector(config)
    if not mt5.connect():
        logger.error("❌ Falha na conexão MT5. Abra o terminal Exness primeiro.")
        sys.exit(1)

    acc = mt5.get_account_info()
    print_account_status(acc)

    # ── Setup componentes ─────────────────────────────────────────────────────
    cache = CacheManager(config)
    dc    = DataCollector(config, mt5)
    dc.set_cache(cache)
    ml    = JLMLModels(config, None)
    risk  = RiskManager(config)

    # ── Verifica modelos ──────────────────────────────────────────────────────
    total_models = len(ml.get_model_list('EUR_USD'))

    if total_models == 0:
        logger.warning("⚠️  Nenhum modelo treinado encontrado!")
        logger.warning("    Execute primeiro: python train_models_live.py")
        logger.info("    Continuando em modo de monitoramento (sem sinais ML)...")
    else:
        logger.info(f"✅ {total_models} modelo(s) carregado(s)")
        logger.info(f"   EUR_USD: {ml.get_model_list('EUR_USD')}")

    # ── Cotações iniciais ─────────────────────────────────────────────────────
    pairs = [('EURUSDm', 'EUR_USD')]
    print("\n  📈 COTAÇÕES AO VIVO:")
    for sym_mt5, sym_key in pairs:
        price = mt5.get_current_price(sym_key) # Usa sym_key porque mt5_connector aplica o sufixo
        print(f"    {sym_mt5}: {price:.5f}" if price else f"    {sym_mt5}: indisponível")

    modo = "CONTA DEMO (EXECUÇÃO REAL)" if not config.is_production() else "⚠️  PRODUÇÃO REAL"
    print(f"\n  🔄 Iniciando loop de análise [{modo}]")
    print("  ⌨️   Pressione Ctrl+C para parar\n")
    print("  " + "-" * 61)

    ciclo = 0
    MAX_CICLOS_SOBREVIVENCIA = 60 # Exatamente 1 Hora Crítica Limite (1 min cada)
    
    try:
        while ciclo < MAX_CICLOS_SOBREVIVENCIA:
            ciclo += 1
            print(f"\n  ⏱  Ciclo #{ciclo}/{MAX_CICLOS_SOBREVIVENCIA} — {datetime.now().strftime('%H:%M:%S')}")

            # 1. GERENCIAR POSIÇÕES EXISTENTES ANTES DE ABRIR NOVAS (CUT LOSS / TAKE PROFIT)
            rs = risk.get_status()
            open_positions = mt5.get_open_positions()
            
            pnl_live = 0.0
            print(f"\n  💼 POSIÇÕES ABERTAS:")
            if open_positions:
                for pos in open_positions:
                    pct_profit = (pos['profit'] / acc['balance'] * 100) if acc and acc['balance'] > 0 else 0
                    icon = "🟢" if pos['profit'] > 0 else "🔴"
                    print(f"    {icon} {pos['symbol']} | {pos['type']} {pos['volume']} lotes | "
                          f"Entrada: {pos['open_price']:.5f} | Atual: {pos['current_price']:.5f} | "
                          f"Lucro: ${pos['profit']:.2f} ({pct_profit:.2f}%)")
                    pnl_live += pos['profit']

                    # Auto Take-Profit (Logica de fechar se estiver ganhando)
                    if pos['profit'] >= 1.00:
                        import MetaTrader5 as mql5
                        logger.info(f"✨ TAKE PROFIT ATIVO! Fechando Ticket {pos['ticket']} garantindo Lucro: ${pos['profit']:.2f}")
                        tick = mql5.symbol_info_tick(pos['symbol'])
                        if tick:
                            close_type = mql5.ORDER_TYPE_BUY if pos['type'] == 'SELL' else mql5.ORDER_TYPE_SELL
                            close_price = tick.ask if close_type == mql5.ORDER_TYPE_BUY else tick.bid
                            
                            close_req = {
                                'action': mql5.TRADE_ACTION_DEAL,
                                'symbol': pos['symbol'],
                                'volume': pos['volume'],
                                'type': close_type,
                                'position': pos['ticket'],
                                'price': close_price,
                                'deviation': 20,
                                'magic': 234000,
                                'comment': 'Auto TP',
                                'type_time': mql5.ORDER_TIME_GTC,
                                'type_filling': mql5.ORDER_FILLING_IOC,
                            }
                            res = mql5.order_send(close_req)
                            if res and res.retcode == mql5.TRADE_RETCODE_DONE:
                                logger.info(f"✅ Lucro Colhido com Sucesso no Ticket {pos['ticket']}.")
                            else:
                                logger.error(f"❌ Erro ao realizar Take Profit no Ticket {pos['ticket']}")
            else:
                print("    Nenhuma posição aberta no momento.")

            # Auto Cut-Loss Único: Fecha apenas a posição com O MAIOR PREJUÍZO (se houver) a cada ciclo
            if open_positions:
                current_open = mt5.get_open_positions() # Recarrega posições após os TPs
                if current_open:
                    worst_pos = min(current_open, key=lambda x: x['profit'])
                    # Ignorar zero e cêntimos residuais flutuantes negativos para manter ordem "0.00" aberta.
                    if worst_pos['profit'] <= -0.01:
                        import MetaTrader5 as mql5
                        logger.warning(f"✂️ CORTANDO PIOR PERDA! Fechando Ticket {worst_pos['ticket']} com Prejuízo: ${worst_pos['profit']:.2f}")
                        tick = mql5.symbol_info_tick(worst_pos['symbol'])
                        if tick:
                            close_type = mql5.ORDER_TYPE_BUY if worst_pos['type'] == 'SELL' else mql5.ORDER_TYPE_SELL
                            close_price = tick.ask if close_type == mql5.ORDER_TYPE_BUY else tick.bid
                            
                            close_req = {
                                'action': mql5.TRADE_ACTION_DEAL,
                                'symbol': worst_pos['symbol'],
                                'volume': worst_pos['volume'],
                                'type': close_type,
                                'position': worst_pos['ticket'],
                                'price': close_price,
                                'deviation': 20,
                                'magic': 234000,
                                'comment': 'Worst Cut-Loss',
                                'type_time': mql5.ORDER_TIME_GTC,
                                'type_filling': mql5.ORDER_FILLING_IOC,
                            }
                            res = mql5.order_send(close_req)
                            if res and res.retcode == mql5.TRADE_RETCODE_DONE:
                                logger.info(f"✅ Pior Perda Cortada com Sucesso no Ticket {worst_pos['ticket']}.")
                            else:
                                logger.error(f"❌ Erro ao relizar Cut-Loss no Ticket {worst_pos['ticket']}")

            # Proteção de Nível de Margem (Circuit Breaker Secundário)
            acc_refresh = mt5.get_account_info()
            if acc_refresh and acc_refresh.get('margin_level'):
                margin_level = acc_refresh['margin_level']
                print(f"  📊 Nível de Margem: {margin_level:.2f}%")
                
                if margin_level < 120.0 and open_positions:
                    logger.critical(f"🚨 ALERTA DE MARGEM ({margin_level:.2f}% < 120%). Fechando piores posições...")
                    
                    # Ordena do maior prejuízo para o menor prejuízo
                    sorted_positions = sorted(open_positions, key=lambda x: x['profit'])
                    
                    import MetaTrader5 as mql5
                    for worst_pos in sorted_positions:
                        logger.warning(f"  ✂️ Fechando Ticket {worst_pos['ticket']} (Prejuízo: ${worst_pos['profit']})")
                        
                        tick = mql5.symbol_info_tick(worst_pos['symbol'])
                        if tick:
                            close_type = mql5.ORDER_TYPE_BUY if worst_pos['type'] == 'SELL' else mql5.ORDER_TYPE_SELL
                            close_price = tick.ask if close_type == mql5.ORDER_TYPE_BUY else tick.bid
                            
                            close_req = {
                                'action': mql5.TRADE_ACTION_DEAL,
                                'symbol': worst_pos['symbol'],
                                'volume': worst_pos['volume'],
                                'type': close_type,
                                'position': worst_pos['ticket'],
                                'price': close_price,
                                'deviation': 20,
                                'magic': 234000,
                                'comment': 'Close by Margin',
                                'type_time': mql5.ORDER_TIME_GTC,
                                'type_filling': mql5.ORDER_FILLING_IOC,
                            }
                            res = mql5.order_send(close_req)
                            if res and res.retcode == mql5.TRADE_RETCODE_DONE:
                                logger.info(f"✅ Ticket {worst_pos['ticket']} Fechado com Sucesso.")
                            else:
                                logger.error(f"❌ Erro ao fechar Ticket {worst_pos['ticket']}")
                        
                        # Re-checa margem antes de fechar a próxima
                        acc_check = mt5.get_account_info()
                        if acc_check and acc_check.get('margin_level', 0) > 120.0:
                            logger.info(f"🛡 Margem recuperada ({acc_check['margin_level']:.2f}%). Parando cortes.")
                            break
            
            # Status de risco global
            print(f"\n  🛡  Risco Fechado: PnL dia=${rs['daily_pnl']:.2f} | "
                  f"Lucro Flutuante=${pnl_live:.2f}")

            # 2. ANALISAR E ABRIR NOVA POSIÇÃO
            for sym_mt5, sym_key in pairs:
                try:
                    signal = analyze_pair(sym_mt5, sym_key, dc, ml, cache)
                    if signal:
                        print_signal(signal)
                        if signal['action'] != 'HOLD':
                            current_open = mt5.get_open_positions()
                            sym_open = [p for p in current_open if p['symbol'] == sym_mt5] if current_open else []
                            
                            # Bloqueio 1: Limite de posições simultâneas
                            if len(sym_open) >= config.risk.max_positions:
                                print(f"  🛡 Limite de posições ativas ({config.risk.max_positions}) alcançado para {sym_key}.")
                                continue

                            # Bloqueio 2: Evitar loop infinito (1 entrada por vela se o cache estiver off)
                            # Se já temos 1 posição aberta para este par, não abrimos outra no mesmo ciclo
                            if len(sym_open) > 0:
                                print(f"  ⏳ Já existe uma posição aberta para {sym_mt5}. Aguardando conclusão ou nova oportunidade.")
                                continue
                                
                            risk_ok = risk.can_trade(sym_key)
                            if risk_ok:
                                logger.info(f"🚀 EXECUTANDO TRADE na {modo}: {signal['action']} {sym_key}")
                                
                                # Preparar a ordem com risco gerenciado (Lote Fixo Reduzido para Tático)
                                try:
                                    # Gestão Rígida de Capital ($500 Bankroll):
                                    exec_price = mt5.get_current_price(sym_key) or signal['price']
                                    atr = signal.get('atr', 0.00150)
                                    
                                    volume = risk.calculate_position_size(
                                        symbol=sym_key, 
                                        price=exec_price, 
                                        atr=atr, 
                                        account_balance=acc['balance'], 
                                        model_confidence=signal.get('confidence', 0.5)
                                    )
                                    
                                    if volume <= 0:
                                        logger.warning(f"  🛡 Risco bloqueou o trade: O Stop Loss excede o capital seguro de 1% (limite $5) em {sym_key}")
                                        continue
                                    
                                    # Alvos Dinâmicos baseados no ATR para respeitar R:R
                                    atr_sl = max(0.00150, atr * 2.5) # SL dinâmico seguro
                                    atr_tp = max(0.00300, atr * 5.0) # TP de 1:2 R:R mínimo
                                    
                                    if signal['action'] == 'BUY':
                                        sl = round(exec_price - atr_sl, 5)
                                        tp = round(exec_price + atr_tp, 5)
                                    else: # SELL
                                        sl = round(exec_price + atr_sl, 5)
                                        tp = round(exec_price - atr_tp, 5)
                                    
                                    order = {
                                        'symbol': sym_key,
                                        'type': signal['action'],
                                        'volume': volume,
                                        'price': signal['price'],
                                        'stop_loss': sl,
                                        'take_profit': tp,
                                        'comment': f"TATIC_{signal['action']}_{int(signal['confidence']*100)}"
                                    }
                                    
                                    exec_result = mt5.place_order(order)
                                    if exec_result and exec_result.get('success'):
                                        logger.info(f"✅ TRADE EXECUTADO NO MT5: TICKET {exec_result.get('ticket')} (Volume Tático: {volume})")
                                    else:
                                        logger.error(f"❌ Falha ao tentar enviar ordem para MT5: {exec_result.get('error') if exec_result else 'Erro desconhecido'}")
                                except Exception as e:
                                    logger.error(f"❌ Erro na montagem/envio da ordem de trade: {e}")
                            else:
                                logger.warning(f"  🛡 Risco bloqueou trade em {sym_key}")
                    else:
                        print(f"  ⚪ {sym_key:<10} | HOLD | Dados/modelos insuficientes")

                except Exception as e:
                    logger.error(f"  Erro ao analisar {sym_key}: {e}")

            # Aguarda próximo ciclo (60 segundos = 1 minuto)
            if ciclo < MAX_CICLOS_SOBREVIVENCIA:
                print(f"\n  ⏳ Próxima análise em 1 minuto (60s)... (Ciclo {ciclo}/{MAX_CICLOS_SOBREVIVENCIA})")
                time.sleep(60)
            else:
                print(f"\n  🛑 TEMPO MÁXIMO (1 HORA) ATINGIDO. Encerrando e limpando ordens abertas.")
                import MetaTrader5 as mql5
                current_open = mt5.get_open_positions()
                if current_open:
                    print(f"  🧹 Fechando {len(current_open)} posição(ões) aberta(s)...")
                    for pos in current_open:
                        tick = mql5.symbol_info_tick(pos['symbol'])
                        if tick:
                            close_type = mql5.ORDER_TYPE_BUY if pos['type'] == 'SELL' else mql5.ORDER_TYPE_SELL
                            close_price = tick.ask if close_type == mql5.ORDER_TYPE_BUY else tick.bid
                            close_req = {
                                'action': mql5.TRADE_ACTION_DEAL,
                                'symbol': pos['symbol'],
                                'volume': pos['volume'],
                                'type': close_type,
                                'position': pos['ticket'],
                                'price': close_price,
                                'deviation': 20,
                                'magic': 234000,
                                'comment': 'Auto-Close Timeout',
                                'type_time': mql5.ORDER_TIME_GTC,
                                'type_filling': mql5.ORDER_FILLING_IOC,
                            }
                            res = mql5.order_send(close_req)
                            if res and res.retcode == mql5.TRADE_RETCODE_DONE:
                                logger.info(f"✅ Ticket {pos['ticket']} fechado com sucesso pelo limite de tempo.")
                            else:
                                logger.error(f"❌ Erro ao fechar Ticket {pos['ticket']}.")
                break # Sai do loop principal

    except KeyboardInterrupt:
        print("\n\n  ⏹  Parando bot...")
        mt5.disconnect()
        print("  ✅ Bot encerrado com sucesso.\n")


if __name__ == '__main__':
    main()
