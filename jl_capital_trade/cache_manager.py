# =============================================================================
# JL CAPITAL TRADE - GERENCIADOR DE CACHE
# =============================================================================

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import threading
import logging
from typing import Any, Optional, Dict
import time
import redis

logger = logging.getLogger(__name__)

class CacheManager:
    """Gerenciador de cache com expiração automática"""
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache em memória
        self.memory_cache = {}
        self.memory_timestamps = {}
        
        # Redis (opcional)
        self.redis_client = None
        if config.cache.redis_enabled:
            try:
                self.redis_client = redis.Redis(
                    host=config.redis.host,
                    port=config.redis.port,
                    password=config.redis.password,
                    db=config.redis.db,
                    decode_responses=True
                )
                logger.info("✅ Redis cache initialized")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
        
        # Lock para thread safety
        self.lock = threading.Lock()
        
        # Inicia thread de limpeza
        self.start_cleanup_thread()
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Gera chave única baseada nos parâmetros"""
        key_string = prefix + "_" + "_".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str, **kwargs) -> Optional[Any]:
        """Obtém item do cache"""
        if not self.config.cache.enabled:
            return None
            
        cache_key = self._generate_key(key, **kwargs)
        
        # Tenta Redis primeiro
        if self.redis_client:
            try:
                data = self.redis_client.get(cache_key)
                if data:
                    logger.debug(f"Cache HIT (redis): {key}")
                    return pickle.loads(data.encode('latin1'))
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Depois memória
        with self.lock:
            if cache_key in self.memory_cache:
                if self._is_valid(cache_key, self.memory_timestamps.get(cache_key)):
                    logger.debug(f"Cache HIT (memory): {key}")
                    return self.memory_cache[cache_key]
                else:
                    # Expirou, remove da memória
                    self.memory_cache.pop(cache_key, None)
                    self.memory_timestamps.pop(cache_key, None)
        
        # Finalmente disco
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                if self._is_valid(cache_key, data.get('timestamp')):
                    # Carrega para memória
                    with self.lock:
                        self.memory_cache[cache_key] = data['value']
                        self.memory_timestamps[cache_key] = data['timestamp']
                    logger.debug(f"Cache HIT (disk): {key}")
                    return data['value']
                else:
                    cache_file.unlink()
            except Exception as e:
                logger.error(f"Cache read error: {e}")
        
        logger.debug(f"Cache MISS: {key}")
        return None
    
    def set(self, value: Any, key: str, ttl: int = 3600, **kwargs):
        """Armazena item no cache"""
        if not self.config.cache.enabled:
            return
            
        cache_key = self._generate_key(key, **kwargs)
        timestamp = datetime.now() + timedelta(seconds=ttl)
        
        # Salva em Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    pickle.dumps(value).decode('latin1')
                )
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # Salva em memória
        with self.lock:
            self.memory_cache[cache_key] = value
            self.memory_timestamps[cache_key] = timestamp
        
        # Salva em disco
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'timestamp': timestamp,
                    'created_at': datetime.now()
                }, f)
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def _is_valid(self, key: str, timestamp: Optional[datetime]) -> bool:
        """Verifica se o cache ainda é válido"""
        if not timestamp:
            return False
        return datetime.now() < timestamp
    
    def clear_expired(self):
        """Limpa itens expirados"""
        with self.lock:
            # Limpa memória
            expired_memory = [
                k for k, ts in self.memory_timestamps.items() 
                if datetime.now() >= ts
            ]
            for k in expired_memory:
                self.memory_cache.pop(k, None)
                self.memory_timestamps.pop(k, None)
            
            # Limpa disco
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    if datetime.now() >= data['timestamp']:
                        cache_file.unlink()
                except:
                    cache_file.unlink()
    
    def start_cleanup_thread(self):
        """Inicia thread para limpeza automática"""
        def cleanup_loop():
            while True:
                time.sleep(3600)  # Executa a cada hora
                self.clear_expired()
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        memory_size = len(self.memory_cache)
        disk_files = len(list(self.cache_dir.glob("*.pkl")))
        
        redis_info = {}
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
            except:
                redis_info = {'error': 'Could not get Redis info'}
        
        return {
            'memory_entries': memory_size,
            'disk_files': disk_files,
            'redis_enabled': self.redis_client is not None,
            'redis_info': redis_info,
            'cache_dir': str(self.cache_dir)
        }
    
    def clear_all(self):
        """Limpa todo o cache"""
        with self.lock:
            self.memory_cache.clear()
            self.memory_timestamps.clear()
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            if self.redis_client:
                self.redis_client.flushdb()
            
            logger.info("🧹 Cache cleared")