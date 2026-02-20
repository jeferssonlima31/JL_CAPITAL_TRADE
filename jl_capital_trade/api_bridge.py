# =============================================================================
# JL CAPITAL TRADE - API BRIDGE PARA OPENCLAW
# =============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import logging
from .trading_bot import JLTradingBot
from .security import SecurityManager, AuditLogger
from .config import config

app = Flask(__name__)
CORS(app)

# Instância do bot
bot = None
security = SecurityManager(config)
audit = AuditLogger()

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok', 
        'version': '2.0.0',
        'environment': config.environment.value
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint para análise de pares"""
    try:
        data = request.json
        pair = data.get('pair', 'EUR_USD')
        timeframe = data.get('timeframe', 'H1')
        
        # Valida entrada
        if pair not in ['EUR_USD', 'XAU_USD']:
            return jsonify({'error': 'Par inválido'}), 400
        
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 503
        
        # Executa análise
        signal = bot.analyze_pair(pair, timeframe)
        
        # Log para auditoria
        audit.log_action(
            user=request.remote_addr,
            action='analyze',
            resource=pair,
            status='success'
        )
        
        return jsonify(signal if signal else {'error': 'Não foi possível analisar'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/execute', methods=['POST'])
def execute():
    """Endpoint para execução de trades"""
    try:
        data = request.json
        
        # Verifica token de autorização
        token = request.headers.get('X-OpenClaw-Token')
        if not security.verify_jwt_token(token):
            return jsonify({'error': 'Não autorizado'}), 401
        
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 503
        
        # Valida dados
        required = ['symbol', 'action', 'price']
        if not all(k in data for k in required):
            return jsonify({'error': 'Dados incompletos'}), 400
        
        # Executa trade
        result = bot.execute_trade(data)
        
        audit.log_action(
            user=request.remote_addr,
            action='execute',
            resource=data.get('symbol', 'unknown'),
            status='success',
            details=data
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Status do bot"""
    try:
        if not bot:
            return jsonify({'status': 'stopped', 'bot': None})
        
        return jsonify({
            'status': 'running',
            'bot': bot.get_status()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def models():
    """Informações dos modelos ML"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 400
        
        return jsonify({
            'eur_usd': bot.ml_models.get_model_list('EUR_USD'),
            'xau_usd': bot.ml_models.get_model_list('XAU_USD'),
            'weights': bot.continuous_learner.tracker.get_model_weights() if bot.continuous_learner else {}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/positions', methods=['GET'])
def positions():
    """Posições abertas"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 400
        
        return jsonify({
            'count': len(bot.positions),
            'positions': list(bot.positions.values())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/risk', methods=['GET'])
def risk():
    """Status de risco"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 400
        
        return jsonify(bot.risk_manager.get_status())
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_api_server(bot_instance=None, port=5000, host='0.0.0.0'):
    """Inicia o servidor API"""
    global bot
    bot = bot_instance
    
    logging.info(f"🚀 API Bridge iniciada em {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    start_api_server()