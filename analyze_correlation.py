#!/usr/bin/env python3
"""
ANÁLISE DE CORRELAÇÃO DAS FEATURES COM O TARGET
"""

import pandas as pd
import numpy as np

# Carrega dados
df = pd.read_csv('trained_models/training_data_enhanced.csv')

# Remove coluna symbol (não numérica)
df_numeric = df.drop(['symbol'], axis=1, errors='ignore')

print('=== CORRELAÇÃO COM TARGET ===')
corr_with_target = df_numeric.corr()['target'].abs().sort_values(ascending=False)
print(corr_with_target.head(15))

print('\n=== FEATURES MAIS CORRELACIONADAS (corr > 0.02) ===')
features_significativas = []
for feature, corr in corr_with_target.items():
    if corr > 0.02 and feature != 'target':
        print(f'{feature}: {corr:.4f}')
        features_significativas.append(feature)

print(f'\nTotal de features significativas: {len(features_significativas)}')

print('\n=== DISTRIBUIÇÃO DO TARGET ===')
print(f'Target=1: {df["target"].sum()} registros')
print(f'Target=0: {len(df) - df["target"].sum()} registros')
print(f'Proporção: {df["target"].mean():.3f}')

# Verifica se há features com correlação decente
if len(features_significativas) == 0:
    print('\nALERTA CRITICO: Nenhuma feature tem correlação significativa com o target!')
    print('Isso explica a baixa acurácia dos modelos.')
else:
    print('\nFeatures com alguma correlação encontradas.')
    print('O problema pode ser no target definition ou vazamento temporal.')