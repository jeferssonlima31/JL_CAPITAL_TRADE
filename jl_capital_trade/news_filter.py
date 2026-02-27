# =============================================================================
# JL CAPITAL TRADE - FILTRO DE NOTÍCIAS ECONÔMICAS
# =============================================================================

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)

class NewsFilter:
    """Filtro de notícias econômicas para evitar volatilidade extrema"""
    
    def __init__(self, config):
        self.config = config
        self.news_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.events: List[Dict] = []
        self.last_update = None
        
        # Parâmetros de Filtro
        self.impact_levels = ['High', 'Medium'] # Níveis de impacto a filtrar
        self.currencies = ['USD', 'EUR']        # Moedas relevantes para EURUSD
        
        # Janelas de Pausa (minutos)
        self.pause_before = int(os.getenv('NEWS_PAUSE_BEFORE', '30'))
        self.pause_after = int(os.getenv('NEWS_PAUSE_AFTER', '60'))
        
        # Cache local
        self.cache_file = config.base_dir / "cache" / "economic_calendar.json"
        
    def update_news(self) -> bool:
        """Atualiza o calendário de notícias"""
        try:
            logger.info("📡 Atualizando calendário de notícias econômicas...")
            response = requests.get(self.news_url, timeout=10)
            
            if response.status_code == 200:
                self.events = response.json()
                self.last_update = datetime.now()
                
                # Salva em cache
                with open(self.cache_file, 'w') as f:
                    import json
                    json.dump(self.events, f)
                
                logger.info(f"✅ Calendário atualizado: {len(self.events)} eventos carregados.")
                return True
            else:
                logger.warning(f"⚠️ Falha ao buscar notícias: Status {response.status_code}")
                return self._load_from_cache()
                
        except Exception as e:
            logger.error(f"❌ Erro ao atualizar notícias: {e}")
            return self._load_from_cache()

    def _load_from_cache(self) -> bool:
        """Carrega notícias do cache local se houver falha na rede"""
        if self.cache_file.exists():
            try:
                import json
                with open(self.cache_file, 'r') as f:
                    self.events = json.load(f)
                logger.info("📁 Notícias carregadas do cache local.")
                return True
            except:
                pass
        return False

    def is_trading_allowed(self) -> Dict:
        """Verifica se o trading é permitido no momento atual baseado nas notícias"""
        if not self.events:
            self.update_news()
            
        now = datetime.utcnow() # Forex Factory usa UTC
        
        upcoming_news = []
        is_allowed = True
        reason = ""
        
        for event in self.events:
            # Filtra por moeda e impacto
            if event['country'] in self.currencies and event['impact'] in self.impact_levels:
                try:
                    # Parse da data (Formato: "MM-DD-YYYY HH:MMam/pm")
                    # Exemplo: "02-27-2026 1:30pm"
                    event_time = datetime.strptime(f"{event['date']} {event['time']}", "%m-%d-%Y %I:%M%p")
                    
                    # Janela de proteção
                    start_pause = event_time - timedelta(minutes=self.pause_before)
                    end_pause = event_time + timedelta(minutes=self.pause_after)
                    
                    if start_pause <= now <= end_pause:
                        is_allowed = False
                        reason = f"Notícia de Alto Impacto: {event['title']} ({event['country']}) às {event['time']}"
                        upcoming_news.append(event)
                        break # Já bloqueou, não precisa continuar
                        
                except Exception as e:
                    continue
                    
        return {
            'allowed': is_allowed,
            'reason': reason,
            'timestamp': now.isoformat(),
            'upcoming_events': upcoming_news
        }

    def check_volatility_protection(self, atr_current: float, atr_mean: float) -> bool:
        """Proteção extra contra volatilidade extrema (anomalias de preço)"""
        if atr_mean > 0:
            volatility_ratio = atr_current / atr_mean
            if volatility_ratio > 2.5: # Volatilidade 2.5x acima da média
                logger.warning(f"🚨 Volatilidade Extrema Detectada! Ratio: {volatility_ratio:.2f}")
                return False
        return True
