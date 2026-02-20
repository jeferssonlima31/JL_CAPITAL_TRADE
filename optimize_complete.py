#!/usr/bin/env python3
"""
OTIMIZAÇÃO COMPLETA DO XGBOOST - SEM EMOJIS
"""

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

print("INICIANDO OTIMIZAÇÃO COMPLETA DO XGBOOST")
print("=" * 60)

# Verifica se os dados existem
data_path = 'trained_models/training_data_enhanced.csv'
if not os.path.exists(data_path):
    print("ERRO: Arquivo de dados não encontrado:", data_path)
    exit(1)

# Carrega dados
print("Carregando dados...")
df = pd.read_csv(data_path)
print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} features")

# Separa features e target
X = df.drop(['target', 'symbol'], axis=1, errors='ignore')
y = df['target']

# Remove NaN
X = X.fillna(X.median())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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

print("Iniciando busca em grade...")

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
    n_jobs=1,  # Evita problemas de paralelismo
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
print("Treinando modelo final...")
final_model = xgb.XGBClassifier(**grid_search.best_params_, random_state=42)
final_model.fit(X, y)

# Salva modelo
model_filename = 'trained_models/xgboost_optimized_complete.model'
final_model.save_model(model_filename)
print(f"Modelo salvo: {model_filename}")

print(f"\nOTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
print(f"Acurácia final: {accuracy:.2%}")