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

    # Coleta dados
    df = dc.get_historical_data(symbol_mt5, 'H1', 500)
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
    lookback = (config.ml.eurusd_lookback
                if symbol_key == 'EUR_USD'
                else config.ml.xauusd_lookback)

    if len(features) < lookback:
        return None

    X = features[-lookback:].reshape(1, lookback, features.shape[1])
    predictions = ml.predict_ensemble(symbol_key, X)

    if not predictions or 'ensemble' not in predictions:
        return None

    ens = float(predictions['ensemble'][0])
    confidence = abs(ens - 0.5) * 2
    current_price = float(df['close'].iloc[-1])
    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0

    if ens > config.ml.buy_threshold and confidence > config.ml.confidence_threshold:
        action = 'BUY'
    elif ens < config.ml.sell_threshold and confidence > config.ml.confidence_threshold:
        action = 'SELL'
    else:
        action = 'HOLD'

    signal = {
        'symbol'      : symbol_key,
        'action'      : action,
        'confidence'  : confidence,
        'ensemble'    : ens,
        'price'       : current_price,
        'atr'         : atr,
        'models_used' : list(predictions.keys()),
        'timestamp'   : datetime.now().isoformat()
    }

    if action != 'HOLD':
        cache.set(signal, cache_key, ttl=900)  # cache por 15 min

    return signal


def print_signal(signal: dict):
    icon  = '🟢' if signal['action'] == 'BUY' else ('🔴' if signal['action'] == 'SELL' else '⚪')
    conf  = signal.get('confidence', 0)
    price = signal.get('price', 0)
    print(f"  {icon} {signal['symbol']:<10} | {signal['action']:<5} | "
          f"Confiança: {conf:.1%} | Preço: {price:.5f} | "
          f"Modelos: {signal.get('models_used', [])}")


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
    total_models = (len(ml.get_model_list('EUR_USD')) +
                    len(ml.get_model_list('XAU_USD')))

    if total_models == 0:
        logger.warning("⚠️  Nenhum modelo treinado encontrado!")
        logger.warning("    Execute primeiro: python train_models_live.py")
        logger.info("    Continuando em modo de monitoramento (sem sinais ML)...")
    else:
        logger.info(f"✅ {total_models} modelo(s) carregado(s)")
        logger.info(f"   EUR_USD: {ml.get_model_list('EUR_USD')}")
        logger.info(f"   XAU_USD: {ml.get_model_list('XAU_USD')}")

    # ── Cotações iniciais ─────────────────────────────────────────────────────
    pairs = [('EURUSD', 'EUR_USD'), ('XAUUSD', 'XAU_USD')]
    print("\n  📈 COTAÇÕES AO VIVO:")
    for sym_mt5, _ in pairs:
        price = mt5.get_current_price(sym_mt5)
        print(f"    {sym_mt5}: {price:.5f}" if price else f"    {sym_mt5}: indisponível")

    modo = "SIMULAÇÃO" if not config.is_production() else "⚠️  PRODUÇÃO REAL"
    print(f"\n  🔄 Iniciando loop de análise [{modo}]")
    print("  ⌨️   Pressione Ctrl+C para parar\n")
    print("  " + "-" * 61)

    ciclo = 0
    try:
        while True:
            ciclo += 1
            print(f"\n  ⏱  Ciclo #{ciclo} — {datetime.now().strftime('%H:%M:%S')}")

            # Analisa cada par
            for sym_mt5, sym_key in pairs:
                try:
                    signal = analyze_pair(sym_mt5, sym_key, dc, ml, cache)
                    if signal:
                        print_signal(signal)
                        if signal['action'] != 'HOLD':
                            risk_ok = risk.can_trade(sym_key)
                            if risk_ok:
                                if config.is_production():
                                    logger.info(f"🚀 EXECUTANDO TRADE REAL: {signal['action']} {sym_key}")
                                    # TODO: bot.execute_trade(signal)
                                else:
                                    logger.info(f"🧪 SIMULAÇÃO — teria executado: {signal['action']} {sym_key} @ {signal['price']:.5f}")
                            else:
                                logger.warning(f"  🛡 Risco bloqueou trade em {sym_key}")
                    else:
                        print(f"  ⚪ {sym_key:<10} | HOLD | Dados/modelos insuficientes")

                except Exception as e:
                    logger.error(f"  Erro ao analisar {sym_key}: {e}")

            # Status de risco
            rs = risk.get_status()
            print(f"\n  🛡  Risco: PnL dia=${rs['daily_pnl']:.2f} | "
                  f"Posições abertas={rs['positions_count']}")

            # Cache stats
            cs = cache.get_stats()
            print(f"  💾 Cache: {cs['memory_entries']} entradas em memória")

            # Aguarda próximo ciclo (60 segundos)
            print(f"\n  ⏳ Próxima análise em 60s... (Ctrl+C para parar)")
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\n  ⏹  Parando bot...")
        mt5.disconnect()
        print("  ✅ Bot encerrado com sucesso.\n")


if __name__ == '__main__':
    main()
