# =============================================================================
# JL CAPITAL TRADE - CONFIGURAÇÕES DO SISTEMA
# =============================================================================

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import logging
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

class Environment(Enum):
    """Ambientes de execução"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Níveis de log"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

@dataclass
class DatabaseConfig:
    """Configurações do banco de dados"""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "jl_capital")
    user: str = os.getenv("DB_USER", "jl_user")
    password: str = os.getenv("DB_PASSWORD", "")
    ssl_mode: str = os.getenv("DB_SSL_MODE", "require")
    
    @property
    def connection_string(self) -> str:
        """String de conexão segura"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.ssl_mode}"

@dataclass
class RedisConfig:
    """Configurações do Redis para cache"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: str = os.getenv("REDIS_PASSWORD", "")
    db: int = int(os.getenv("REDIS_DB", "0"))
    ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    
    @property
    def connection_string(self) -> str:
        """String de conexão segura"""
        if self.password:
            return f"rediss://:{self.password}@{self.host}:{self.port}/{self.db}" if self.ssl else f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

@dataclass
class MT5Config:
    """Configurações do MetaTrader 5"""
    login: int = int(os.getenv("MT5_LOGIN", "0"))
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    path: str = os.getenv("MT5_PATH", "C:\\Program Files\\MetaTrader 5\\terminal64.exe")
    timeout: int = int(os.getenv("MT5_TIMEOUT", "30000"))
    
    def validate(self) -> bool:
        """Valida configurações MT5"""
        return bool(self.login and self.password and self.server)

@dataclass
class OANDAConfig:
    """Configurações da OANDA"""
    enabled: bool = os.getenv("OANDA_ENABLED", "false").lower() == "true"
    api_key: str = os.getenv("OANDA_API_KEY", "")
    account_id: str = os.getenv("OANDA_ACCOUNT_ID", "")
    environment: str = os.getenv("OANDA_ENV", "practice")

@dataclass
class RiskConfig:
    """Configurações de gerenciamento de risco"""
    max_risk_per_trade: float = float(os.getenv("MAX_RISK_PER_TRADE", "1.0"))
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", "3.0"))
    max_positions: int = int(os.getenv("MAX_POSITIONS", "2"))
    default_sl_pips_eurusd: int = int(os.getenv("SL_EURUSD", "15"))
    default_tp_pips_eurusd: int = int(os.getenv("TP_EURUSD", "30"))
    default_sl_pips_xauusd: int = int(os.getenv("SL_XAUUSD", "50"))
    default_tp_pips_xauusd: int = int(os.getenv("TP_XAUUSD", "100"))
    use_trailing_stop: bool = os.getenv("USE_TRAILING_STOP", "false").lower() == "true"
    trailing_activation_pips: int = int(os.getenv("TRAILING_ACTIVATION", "15"))
    max_spread_pips: int = int(os.getenv("MAX_SPREAD_PIPS", "5"))

@dataclass
class MLConfig:
    """Configurações de Machine Learning"""
    # EUR/USD
    eurusd_lookback: int = int(os.getenv("EURUSD_LOOKBACK", "60"))
    eurusd_prediction_horizon: int = int(os.getenv("EURUSD_HORIZON", "5"))
    eurusd_retrain_hours: int = int(os.getenv("EURUSD_RETRAIN", "24"))
    
    # XAU/USD
    xauusd_lookback: int = int(os.getenv("XAUUSD_LOOKBACK", "80"))
    xauusd_prediction_horizon: int = int(os.getenv("XAUUSD_HORIZON", "3"))
    xauusd_retrain_hours: int = int(os.getenv("XAUUSD_RETRAIN", "12"))
    
    # Modelos ativos
    active_models: List[str] = field(default_factory=lambda: os.getenv("ACTIVE_MODELS", "xgboost,lstm,ensemble").split(","))
    
    # Thresholds
    buy_threshold: float = float(os.getenv("BUY_THRESHOLD", "0.65"))
    sell_threshold: float = float(os.getenv("SELL_THRESHOLD", "0.35"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    
    # Model weights (iniciais)
    model_weights: Dict[str, float] = None
    
    def __post_init__(self):
        self.model_weights = {
            'xgboost': 0.30,
            'lstm': 0.35,
            'ensemble': 0.35
        }

@dataclass
class CacheConfig:
    """Configurações de cache"""
    enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    historical_data_ttl: int = int(os.getenv("HIST_DATA_TTL", "86400"))  # 24h
    predictions_ttl: int = int(os.getenv("PRED_TTL", "900"))  # 15min
    max_entries: int = int(os.getenv("MAX_CACHE_ENTRIES", "1000"))
    redis_enabled: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"

class JLConfig:
    """Configuração principal do sistema JL Capital Trade"""
    
    def __init__(self):
        # Ambiente
        self.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = LogLevel[os.getenv("LOG_LEVEL", "INFO")]
        
        # Diretórios
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.reports_dir = self.base_dir / "reports"
        self.backup_dir = self.base_dir / "backups"
        self.cache_dir = self.base_dir / "cache"
        
        # Configurações específicas
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.mt5 = MT5Config()
        self.oanda = OANDAConfig()
        self.risk = RiskConfig()
        self.ml = MLConfig()
        self.cache = CacheConfig()
        
        # Pares de trading
        self.trading_pairs = ["EUR_USD", "XAU_USD"]
        
        # Timeframes
        self.timeframes = {
            "M5": 5,
            "M15": 15,
            "H1": 60,
            "H4": 240,
            "D1": 1440
        }
        
        # Security
        self.encryption_key = os.getenv("ENCRYPTION_KEY", "").encode()
        self.jwt_secret = os.getenv("JWT_SECRET", "")
        self.api_rate_limit = int(os.getenv("API_RATE_LIMIT", "100"))
        
        # API Bridge
        self.api_port = int(os.getenv("API_PORT", "5000"))
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        
        # Criar diretórios
        self._create_directories()
    
    def _create_directories(self):
        """Cria diretórios necessários com permissões adequadas"""
        for directory in [
            self.data_dir, self.models_dir, self.logs_dir,
            self.reports_dir, self.backup_dir, self.cache_dir
        ]:
            directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    
    def is_production(self) -> bool:
        """Verifica se está em produção"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Verifica se está em teste"""
        return self.environment == Environment.TESTING
    
    def validate(self) -> bool:
        """Valida configurações críticas"""
        if self.is_production():
            # Em produção, exigir configurações mais rigorosas
            assert self.mt5.validate(), "MT5 config inválida em produção"
            assert self.encryption_key, "Chave de criptografia necessária em produção"
            assert self.jwt_secret, "JWT secret necessário em produção"
            assert self.database.password, "Senha de banco necessária em produção"
        
        return True

# Instância global da configuração
config = JLConfig()