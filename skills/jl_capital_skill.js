// jl_capital_skill.js - Skill para OpenClaw

module.exports = {
  name: 'jl-capital-trade',
  description: 'JL Capital Trade - Forex ML Bot (EUR/USD & XAU/USD)',
  version: '2.0.0',
  author: 'JL Capital',
  
  // Configuração da skill
  config: {
    enabled: true,
    requiresAuth: true,
    timeout: 30000,
  },
  
  // Comandos que a skill responde
  commands: [
    {
      name: 'analyze',
      description: 'Analisar par (EUR/USD ou XAU/USD)',
      usage: '/jl analyze --pair EUR_USD --timeframe H1',
      handler: 'handleAnalyze'
    },
    {
      name: 'trade',
      description: 'Executar trade',
      usage: '/jl trade --action BUY --symbol EUR_USD --price 1.0892',
      handler: 'handleTrade'
    },
    {
      name: 'status',
      description: 'Status do bot',
      usage: '/jl status',
      handler: 'handleStatus'
    },
    {
      name: 'positions',
      description: 'Ver posições abertas',
      usage: '/jl positions',
      handler: 'handlePositions'
    },
    {
      name: 'risk',
      description: 'Status de risco',
      usage: '/jl risk',
      handler: 'handleRisk'
    },
    {
      name: 'help',
      description: 'Ajuda',
      usage: '/jl help',
      handler: 'handleHelp'
    }
  ],
  
  // Handlers
  handlers: {
    handleAnalyze: async (args, context) => {
      const { pair = 'EUR_USD', timeframe = 'H1' } = args;
      
      // Valida par
      if (!['EUR_USD', 'XAU_USD'].includes(pair)) {
        return { error: 'Par inválido. Use EUR_USD ou XAU_USD' };
      }
      
      // Chama API
      const response = await fetch(`${context.config.api_url}/analyze`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-OpenClaw-Token': context.config.auth_token
        },
        body: JSON.stringify({ pair, timeframe })
      });
      
      const data = await response.json();
      
      if (data.error) {
        return { error: data.error };
      }
      
      // Formata resposta
      let emoji = '⚪';
      if (data.action === 'BUY') emoji = '🟢';
      if (data.action === 'SELL') emoji = '🔴';
      
      return {
        text: `${emoji} *${pair}* - ${data.action} ${data.strength}\n` +
              `Preço: $${data.price.toFixed(4)}\n` +
              `Confiança: ${(data.confidence * 100).toFixed(1)}%\n` +
              `Stop Loss: $${data.stop_loss.toFixed(4)}\n` +
              `Take Profit: $${data.take_profit.toFixed(4)}`,
        data: data
      };
    },
    
    handleTrade: async (args, context) => {
      const { action, symbol, price, stop_loss, take_profit } = args;
      
      // Validações
      if (!['BUY', 'SELL'].includes(action?.toUpperCase())) {
        return { error: 'Ação inválida. Use BUY ou SELL' };
      }
      
      if (!symbol || !['EUR_USD', 'XAU_USD'].includes(symbol)) {
        return { error: 'Símbolo inválido' };
      }
      
      if (!price || isNaN(price)) {
        return { error: 'Preço inválido' };
      }
      
      // Chama API
      const response = await fetch(`${context.config.api_url}/execute`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-OpenClaw-Token': context.config.auth_token
        },
        body: JSON.stringify({
          symbol,
          action: action.toUpperCase(),
          price: parseFloat(price),
          stop_loss: stop_loss ? parseFloat(stop_loss) : 0,
          take_profit: take_profit ? parseFloat(take_profit) : 0
        })
      });
      
      const data = await response.json();
      
      if (data.error) {
        return { error: data.error };
      }
      
      return {
        text: `✅ Trade executado: ${action} ${symbol} @ $${price}`,
        data: data
      };
    },
    
    handleStatus: async (args, context) => {
      const response = await fetch(`${context.config.api_url}/status`, {
        headers: { 'X-OpenClaw-Token': context.config.auth_token }
      });
      
      const data = await response.json();
      
      if (data.error) {
        return { error: data.error };
      }
      
      if (!data.bot) {
        return { text: '🤖 Bot não está rodando' };
      }
      
      const bot = data.bot;
      const winRate = bot.performance.total_trades > 0 
        ? (bot.performance.winning_trades / bot.performance.total_trades * 100).toFixed(1)
        : 0;
      
      return {
        text: `🤖 *JL Capital Trade Status*\n` +
              `Status: ${data.status === 'running' ? '▶️ Rodando' : '⏹️ Parado'}\n` +
              `Posições: ${bot.positions}\n` +
              `Trades: ${bot.performance.total_trades} (Win: ${winRate}%)\n` +
              `P&L Total: $${bot.performance.total_pnl.toFixed(2)}\n` +
              `MT5: ${bot.mt5_connected ? '✅' : '❌'}\n` +
              `Modelos: ${bot.models_loaded}`,
        data: bot
      };
    },
    
    handlePositions: async (args, context) => {
      const response = await fetch(`${context.config.api_url}/positions`, {
        headers: { 'X-OpenClaw-Token': context.config.auth_token }
      });
      
      const data = await response.json();
      
      if (data.error) {
        return { error: data.error };
      }
      
      if (data.count === 0) {
        return { text: '📭 Nenhuma posição aberta' };
      }
      
      let text = `📊 *Posições Abertas (${data.count})*\n\n`;
      
      data.positions.forEach(pos => {
        const emoji = pos.action === 'BUY' ? '🟢' : '🔴';
        text += `${emoji} *${pos.symbol}* - ${pos.action}\n`;
        text += `   Entrada: $${pos.open_price.toFixed(4)}\n`;
        text += `   Atual: $${pos.price.toFixed(4)}\n`;
        text += `   SL: $${pos.stop_loss.toFixed(4)} | TP: $${pos.take_profit.toFixed(4)}\n`;
        text += `   Confiança: ${(pos.confidence * 100).toFixed(1)}%\n\n`;
      });
      
      return { text, data: data.positions };
    },
    
    handleRisk: async (args, context) => {
      const response = await fetch(`${context.config.api_url}/risk`, {
        headers: { 'X-OpenClaw-Token': context.config.auth_token }
      });
      
      const data = await response.json();
      
      if (data.error) {
        return { error: data.error };
      }
      
      return {
        text: `📊 *Status de Risco*\n` +
              `P&L Diário: $${data.daily_pnl.toFixed(2)}\n` +
              `Posições: ${data.positions_count}\n` +
              `Última atualização: ${new Date(data.last_update).toLocaleString()}`,
        data: data
      };
    },
    
    handleHelp: async (args, context) => {
      return {
        text: `🤖 *JL Capital Trade - Comandos*\n\n` +
              `/jl analyze --pair EUR_USD --timeframe H1\n` +
              `/jl trade --action BUY --symbol EUR_USD --price 1.0892\n` +
              `/jl status\n` +
              `/jl positions\n` +
              `/jl risk\n` +
              `/jl help`
      };
    }
  }
};