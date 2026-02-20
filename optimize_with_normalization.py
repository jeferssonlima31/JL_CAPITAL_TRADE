#!/usr/bin/env python3
"""
OTIMIZAÇÃO COM NORMALIZAÇÃO DAS FEATURES
"""

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

print("INICIANDO OTIMIZAÇÃO COM NORMALIZAÇÃO")
print("=" * 60)

# Carrega dados
df = pd.read_csv('trained_models/training_data_enhanced.csv')
print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} features")

# Separa features e target
X = df.drop(['target', 'symbol'], axis=1, errors='ignore')
y = df['target']

# Remove NaN
X = X.fillna(X.median())

# NORMALIZAÇÃO DAS FEATURES
print("Aplicando normalização das features...")
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

print("Estatísticas após normalização:")
print(f"Médias: {X_normalized.mean().mean():.6f}")
print(f"Desvio padrão: {X_normalized.std().mean():.6f}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Split realizado: Treino {X_train.shape}, Teste {X_test.shape}")

# Parâmetros para otimização
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

print("Iniciando busca em grade com dados normalizados...")

# Grid Search
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Busca em grade concluída!")
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor acurácia (validação): {grid_search.best_score_:.4f}")

# Avaliação no teste
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {accuracy:.4f}")

# Treina modelo final com todos os dados
print("Treinando modelo final com todos os dados...")
final_model = xgb.XGBClassifier(**grid_search.best_params_, random_state=42)
final_model.fit(X_normalized, y)

# Salva modelo e scaler
model_filename = 'trained_models/xgboost_optimized_normalized.model'
final_model.save_model(model_filename)

# Salva o scaler
import joblib
joblib.dump(scaler, 'trained_models/feature_scaler_optimized.pkl')

print(f"Modelo salvo: {model_filename}")
print(f"Scaler salvo: trained_models/feature_scaler_optimized.pkl")

print(f"\nOTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
print(f"Acurácia final: {accuracy:.2%}")