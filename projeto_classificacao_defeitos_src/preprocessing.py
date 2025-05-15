import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

def create_preprocessing_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
