# =============================================================================
# JL CAPITAL TRADE - MODELOS DE MACHINE LEARNING
# Compatível com Python 3.13+ (sem TensorFlow)
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

# XGBoost - importação condicional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("⚠️ XGBoost não disponível")

# Scikit-learn MLP — substituto do LSTM, compatível com Python 3.13
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️ Scikit-learn não disponível")


class JLMLModels:
    """Modelos de ML com capacidade de aprendizado contínuo
    
    Modelos disponíveis:
      - xgboost : XGBoost (gradient boosting)
      - mlp     : MLPClassifier (substituto do LSTM — rede neural totalmente conectada
                  treinada sobre janela de lookback features achatadas)
    """

    def __init__(self, config, continuous_learner=None):
        self.config = config
        self.continuous_learner = continuous_learner
        self.models_dir = config.models_dir

        # Cache de modelos em memória
        self.models: Dict[str, Dict] = {
            'EUR_USD': {},
            'XAU_USD': {}
        }

        # Versões dos modelos
        self.model_versions: Dict[str, str] = {}

        # Scalers necessários para o MLP
        self.scalers: Dict[str, Optional[StandardScaler]] = {
            'EUR_USD': None
        }

        # Modelos Online (SGDClassifier)
        self.online_learners: Dict[str, Optional[SGDClassifier]] = {
            'EUR_USD': self.create_online_learner()
        }

        # Carrega modelos existentes
        self._load_all_models()

        logger.info("🤖 ML Models (XGBoost + MLP) initialized — TF-free ✅")

    # -------------------------------------------------------------------------
    # Carregamento / salvamento
    # -------------------------------------------------------------------------

    def _load_all_models(self):
        """Carrega todos os modelos salvos (.pkl e .joblib)"""
        # Carrega modelos da raiz da pasta trained_models também
        for model_file in self.models_dir.glob("*.joblib"):
            name = model_file.stem
            try:
                obj = joblib.load(model_file)
                # Associa a EUR_USD por padrão se for o modelo agressivo
                if "aggressive" in name or "EURUSD" in name:
                    self.models['EUR_USD'][name] = obj
                logger.info(f"✅ Modelo '{name}' carregado da raiz")
            except Exception as e:
                logger.error(f"Erro ao carregar {model_file}: {e}")

        for symbol in ['EUR_USD', 'XAU_USD']:
            symbol_dir = self.models_dir / symbol
            if not symbol_dir.exists():
                continue

            # .pkl  (pickle padrão)
            for model_file in symbol_dir.glob("*.pkl"):
                name = model_file.stem.split('_v')[0]
                try:
                    with open(model_file, 'rb') as f:
                        obj = pickle.load(f)
                    if name.startswith('scaler'):
                        self.scalers[symbol] = obj
                        logger.info(f"✅ Scaler carregado para {symbol}")
                    else:
                        self.models[symbol][name] = obj
                        logger.info(f"✅ Modelo '{name}' carregado para {symbol}")
                except Exception as e:
                    logger.error(f"Erro ao carregar {model_file}: {e}")

            # .joblib (scikit-learn/xgboost)
            if SKLEARN_AVAILABLE:
                for model_file in symbol_dir.glob("*.joblib"):
                    name = model_file.stem.split('_v')[0]
                    try:
                        obj = joblib.load(model_file)
                        self.models[symbol][name] = obj
                        logger.info(f"✅ Modelo '{name}' (.joblib) carregado para {symbol}")
                    except Exception as e:
                        logger.error(f"Erro ao carregar {model_file}: {e}")

    def save_model(self, symbol: str, model_name: str, model, scaler=None):
        """Salva modelo (e opcionalmente scaler) em disco"""
        symbol_dir = self.models_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        version = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            if SKLEARN_AVAILABLE:
                model_path = symbol_dir / f"{model_name}_v{version}.joblib"
                joblib.dump(model, model_path)
            else:
                model_path = symbol_dir / f"{model_name}_v{version}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            self.models[symbol][model_name] = model
            logger.info(f"✅ Modelo '{model_name}' salvo para {symbol}")

            if scaler is not None:
                scaler_path = symbol_dir / f"scaler_{model_name}_v{version}.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                self.scalers[symbol] = scaler
                logger.info(f"✅ Scaler salvo para {symbol}")

        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")

    # -------------------------------------------------------------------------
    # Engenharia de features
    # -------------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
        """Prepara features para ML — retorna array (n_amostras, n_features)"""
        
        # Se for o modelo agressivo, usamos o AggressiveFeatureCompatibility
        try:
            from aggressive_feature_compatibility import AggressiveFeatureCompatibility
            compat = AggressiveFeatureCompatibility()
            # O conversor agressivo já retorna o formato correto (1, 43)
            # Mas aqui o sistema espera o histórico completo para o lookback
            # Vamos calcular as features para o df inteiro
            
            result_df = df.copy()
            compat._calculate_all_features(result_df)
            return result_df[compat.required_features].values
        except ImportError:
            logger.warning("AggressiveFeatureCompatibility não encontrado, usando features padrão")

        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position', 'atr', 'volume_ratio',
            'close_position', 'volatility', 'ema_cross',
            'momentum', 'roc', 'hurst', 'fractal_dim', 'efficiency_ratio'
        ]

        available = [c for c in feature_columns if c in df.columns]
        if not available:
            logger.error(f"Nenhuma feature disponível para {symbol}")
            return None

        features = df[available].copy()
        features = features.ffill().fillna(0)
        return features.values

    def flatten_lookback(self, features: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converte array (n, n_features) em (n-lookback, lookback*n_features).
        Usado para alimentar MDPs/XGBoost com janela temporal — equivalente ao LSTM.
        Retorna (X_flat, indices_originais).
        """
        X = []
        idx = []
        for i in range(lookback, len(features)):
            window = features[i - lookback:i].flatten()
            X.append(window)
            idx.append(i)
        return np.array(X), np.array(idx)

    # -------------------------------------------------------------------------
    # Criação de modelos
    # -------------------------------------------------------------------------

    def create_xgboost_model(self) -> Optional[object]:
        """Cria modelo XGBoost"""
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost não disponível")
            return None
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

    def create_mlp_model(self) -> Optional[object]:
        """
        Cria MLPClassifier como substituto do LSTM.
        Arquitetura: 128 → 64 → 32 neurônios com relu, adam optimizer.
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn não disponível")
            return None
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False
        )

    def create_online_learner(self) -> Optional[SGDClassifier]:
        """Cria modelo SGDClassifier para Online Learning (partial_fit)"""
        if not SKLEARN_AVAILABLE:
            return None
        return SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            l1_ratio=0.15,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-3,
            shuffle=True,
            verbose=0,
            epsilon=0.1,
            n_jobs=-1,
            random_state=42,
            learning_rate='optimal',
            eta0=0.0,
            power_t=0.5,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            class_weight=None,
            warm_start=False,
            average=False
        )

    def partial_fit_online(self, symbol: str, X: np.ndarray, y: np.ndarray):
        """Atualiza o modelo online com novos dados (Online Learning)"""
        if self.online_learners[symbol] is not None:
            # Garante que as classes sejam [0, 1]
            classes = np.array([0, 1])
            self.online_learners[symbol].partial_fit(X, y, classes=classes)
            logger.info(f"⚡ Online Learner updated for {symbol} (SGD partial_fit)")

    def predict_regime_aware(self, symbol: str, X: np.ndarray, regime: str) -> Dict:
        """Faz previsão adaptada ao regime de mercado (Tendência vs Range)"""
        
        # 1. Faz previsões base
        preds = self.predict_ensemble(symbol, X)
        
        # 2. Ajusta pesos baseados no regime
        # Em 'trending', damos mais peso ao XGBoost (melhor para tendências)
        # Em 'ranging', damos mais peso ao ExtraTrees ou MLP (melhor para reversões)
        
        if regime == "trending":
            # Damos 70% de peso ao modelo principal
            pass
        elif regime == "ranging":
            # Reduzimos threshold ou mudamos pesos
            pass
            
        return preds

    # -------------------------------------------------------------------------
    # Previsão Ensemble
    # -------------------------------------------------------------------------

    def predict_ensemble(self, symbol: str, X: np.ndarray,
                         use_weights: bool = True, regime: str = "normal") -> Dict:
        """
        Faz previsão ensemble combinando modelos estáticos e o online learner.
        """
        results = {}
        probs = []
        weights_list = []

        if symbol not in self.models:
            return results

        # 1. Previsão dos modelos estáticos
        for name, model in self.models[symbol].items():
            try:
                # Determinar formato de entrada correto
                # Modelos "aggressive" ou ExtraTrees/XGBoost básicos esperam 1 bar (n_features)
                # Modelos MDP/LSTM esperam lookback (lookback * n_features)
                
                n_features_total = X.reshape(X.shape[0], -1).shape[1]
                
                # Se X for 3D (batch, lookback, features), pegamos a última bar para modelos single-bar
                if len(X.shape) == 3:
                    X_single_bar = X[:, -1, :] # Última bar: (batch, n_features)
                    X_flattened = X.reshape(X.shape[0], -1) # Flattened: (batch, lookback * n_features)
                else:
                    X_single_bar = X
                    X_flattened = X

                # Tenta detectar qual entrada o modelo espera
                if hasattr(model, "n_features_in_"):
                    expected = model.n_features_in_
                    if expected == X_single_bar.shape[1]:
                        X_input = X_single_bar
                    elif expected == 6 and X_single_bar.shape[1] == 43:
                        # Caso especial: modelo robusto (6) com entrada completa (43)
                        # As features robustas estão em posições específicas
                        from aggressive_feature_compatibility import AggressiveFeatureCompatibility
                        compat = AggressiveFeatureCompatibility()
                        robust_indices = [compat.all_features.index(f) for f in compat.robust_features]
                        X_input = X_single_bar[:, robust_indices]
                    else:
                        X_input = X_flattened
                elif "aggressive" in name.lower():
                    X_input = X_single_bar
                else:
                    X_input = X_flattened

                # Aplicar scaler se existir
                if (name in ('mlp', 'mlp_eurusd', 'mlp_xauusd') or "aggressive" in name) and self.scalers.get(symbol):
                    if "aggressive" in name:
                        for scaler_file in self.models_dir.glob("scaler_aggressive_*.pkl"):
                            try:
                                agg_scaler = joblib.load(scaler_file)
                                X_input = agg_scaler.transform(X_input)
                                break
                            except: continue
                    else:
                        X_input = self.scalers[symbol].transform(X_input)

                if hasattr(model, 'predict_proba'):
                    p = model.predict_proba(X_input)[:, 1]
                else:
                    p = model.predict(X_input).astype(float)
                
                results[name] = p
                
                # Obtém peso do tracker
                weight = 1.0
                if use_weights and self.continuous_learner:
                    weight = self.continuous_learner.tracker.get_model_weight(symbol, name)
                
                # Ajuste de peso por Regime
                if regime == "trending" and ("xgboost" in name.lower()):
                    weight *= 1.5 # Favorece XGBoost em tendência
                elif regime == "ranging" and ("mlp" in name.lower()):
                    weight *= 1.5 # Favorece MLP em range
                
                probs.append(p)
                weights_list.append(weight)
            except Exception as e:
                logger.error(f"Erro na previsão do modelo {name}: {e}")

        # 2. Previsão do Online Learner (SGD)
        online_model = self.online_learners.get(symbol)
        if online_model is not None and hasattr(online_model, "predict_proba"):
            try:
                # Garantir formato 2D
                X_input = X.reshape(X.shape[0], -1) if len(X.shape) == 3 else X
                p_online = online_model.predict_proba(X_input)[:, 1]
                results['online_sgd'] = p_online
                
                # Online learner tem peso fixo inicial ou adaptativo
                weight_online = 0.2
                probs.append(p_online)
                weights_list.append(weight_online)
            except:
                pass

        # 3. Calcula Média Ponderada (Ensemble)
        if probs:
            probs_arr = np.array(probs)
            weights_arr = np.array(weights_list)
            if weights_arr.sum() > 0:
                weights_arr = weights_arr / weights_arr.sum()
                ensemble_prob = np.average(probs_arr, axis=0, weights=weights_arr)
                results['ensemble'] = ensemble_prob
            
        return results

    # -------------------------------------------------------------------------
    # Utilitários
    # -------------------------------------------------------------------------

    def get_model_list(self, symbol: str) -> List[str]:
        """Retorna lista de modelos disponíveis"""
        return list(self.models.get(symbol, {}).keys())

    def create_sequences(self, features: np.ndarray, lookback: int,
                         horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compatibilidade retroativa — retorna (X_flat, [])"""
        X, _ = self.flatten_lookback(features, lookback)
        return X, np.array([])