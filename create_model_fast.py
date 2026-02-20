#!/usr/bin/env python3
"""
CRIA MODELO XGBOOST RÁPIDO E SIMPLES
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

print("🚀 CRIANDO MODELO XGBOOST OTIMIZADO...")

# Carrega dados
df = pd.read_csv('trained_models/training_data_enhanced.csv')
print(f"📊 Dados carregados: {df.shape[0]} registros, {df.shape[1]} features")

# Separa features e target
X = df.drop(['target', 'symbol'], axis=1, errors='ignore')
y = df['target']

# Remove NaN
X = X.fillna(X.median())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🔪 Split: Treino {X_train.shape}, Teste {X_test.shape}")

# Parâmetros otimizados
params = {
    'max_depth': 7,
    'learning_rate': 0.08, 
    'n_estimators': 200,
    'colsample_bytree': 0.8,
    'subsample': 0.9
}

# Cria e treina modelo
model = xgb.XGBClassifier(**params, random_state=42)
print("⚡ Treinando modelo...")
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 ACURÁCIA: {accuracy:.4f} ({accuracy:.2%})")

# Treina com todos os dados
final_model = xgb.XGBClassifier(**params, random_state=42)
final_model.fit(X, y)

# Salva modelo
final_model.save_model('trained_models/xgboost_optimized_final.model')
print("✅ MODELO SALVO: trained_models/xgboost_optimized_final.model")

print(f"\n🚀 SISTEMA PRONTO! Acurácia: {accuracy:.2%}")
print("📈 Execute: python simple_ensemble.py para backtesting")