"""
Funções auxiliares para carregamento e transformação dos dados.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def convert_to_binary(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ['true', '1', 't', 'sim', 'yes']:
            return 1
        elif value in ['false', '0', 'f', 'não', 'no']:
            return 0
        return np.nan
    return int(bool(value))

def carregar_dados(path):
    df = pd.read_csv(path)

    binary_cols = ['tipo_do_aço_A300', 'tipo_do_aço_A400']
    falhas = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']

    for col in binary_cols + falhas:
        df[col] = df[col].apply(convert_to_binary)

    df = df.dropna(subset=binary_cols + falhas)
    df = df[df[falhas].sum(axis=1) == 1].copy()
    df['classe'] = df[falhas].idxmax(axis=1)
    df.drop(columns=falhas, inplace=True)

    le = LabelEncoder()
    df['classe'] = le.fit_transform(df['classe'])

    return df