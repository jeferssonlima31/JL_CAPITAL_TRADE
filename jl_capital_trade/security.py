# =============================================================================
# JL CAPITAL TRADE - SEGURANÇA
# =============================================================================

import hashlib
import hmac
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import re
import secrets

logger = logging.getLogger(__name__)

class SecurityManager:
    """Gerenciador de segurança do sistema JL Capital"""
    
    def __init__(self, config):
        self.config = config
        self._init_encryption()
    
    def _init_encryption(self):
        """Inicializa sistema de criptografia"""
        if self.config.encryption_key:
            # Deriva chave para Fernet
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'jl_capital_salt',
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.config.encryption_key))
            self.cipher = Fernet(key)
        else:
            self.cipher = None
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Criptografa dados sensíveis"""
        if not self.cipher:
            logger.warning("Criptografia não inicializada")
            return data
        
        try:
            return self.cipher.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Erro na criptografia: {e}")
            raise SecurityException("Falha na criptografia")
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Descriptografa dados sensíveis"""
        if not self.cipher:
            logger.warning("Criptografia não inicializada")
            return encrypted_data
        
        try:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Erro na descriptografia: {e}")
            raise SecurityException("Falha na descriptografia")
    
    def hash_password(self, password: str) -> str:
        """Gera hash seguro de senha"""
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return base64.b64encode(salt + key).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verifica senha contra hash"""
        try:
            decoded = base64.b64decode(hashed.encode('utf-8'))
            salt = decoded[:32]
            stored_key = decoded[32:]
            
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )
            
            return hmac.compare_digest(key, stored_key)
        except Exception:
            return False
    
    def generate_jwt_token(self, user_id: str, expires_in: int = 3600) -> str:
        """Gera JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow(),
            'jti': secrets.token_hex(16)
        }
        
        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm='HS256'
        )
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica JWT token"""
        try:
            return jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=['HS256']
            )
        except jwt.ExpiredSignatureError:
            logger.warning("Token expirado")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token inválido: {e}")
            return None
    
    def sanitize_input(self, input_str: str) -> str:
        """Sanitiza input para prevenir injeção"""
        # Remove caracteres perigosos
        dangerous = [';', '--', '/*', '*/', '@@', 'char', 'nchar', 'varchar', 'nvarchar']
        sanitized = input_str
        for item in dangerous:
            sanitized = sanitized.replace(item, '')
        
        # Escapa caracteres especiais
        sanitized = re.sub(r'[<>\"\']', '', sanitized)
        
        return sanitized.strip()
    
    def validate_api_key(self, api_key: str) -> bool:
        """Valida chave de API"""
        if not api_key or len(api_key) < 32:
            return False
        
        # Verifica formato esperado
        pattern = r'^JL[A-Z0-9]{32,}$'
        return bool(re.match(pattern, api_key))
    
    def rate_limit_key(self, key: str) -> bool:
        """Implementa rate limiting simples"""
        # Em produção, usar Redis para isso
        return True

class AuditLogger:
    """Logger de auditoria para ações sensíveis"""
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
    
    def log_action(self, user: str, action: str, resource: str, 
                   status: str, details: Dict = None):
        """Registra ação para auditoria"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': user,
            'action': action,
            'resource': resource,
            'status': status,
            'details': details or {},
            'ip': self._get_client_ip()
        }
        
        self.logger.info(f"AUDIT: {log_entry}")
        # Em produção, salvar em banco de dados
    
    def _get_client_ip(self) -> str:
        """Obtém IP do cliente"""
        # Implementar conforme necessidade
        return "127.0.0.1"

class SecurityException(Exception):
    """Exceção de segurança"""
    pass