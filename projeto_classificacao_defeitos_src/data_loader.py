import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from .preprocessing import convert_to_binary

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    binary_cols = ['tipo_do_aço_A300', 'tipo_do_aço_A400']
    falhas = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']

    for col in binary_cols + falhas:
        df[col] = df[col].apply(convert_to_binary)

    df[binary_cols + falhas] = SimpleImputer(strategy='most_frequent').fit_transform(df[binary_cols + falhas])

    df = df[df[falhas].sum(axis=1) == 1].copy()
    df['classe'] = df[falhas].idxmax(axis=1)
    df.drop(columns=falhas, inplace=True)

    le = LabelEncoder()
    df['classe'] = le.fit_transform(df['classe'])

    X = df.drop(columns=['id', 'classe'])
    y = df['classe']

    return X, y
