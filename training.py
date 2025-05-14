"""
Treinamento do modelo e validação com GridSearch.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
import joblib
from .preprocessing import build_preprocessing_pipeline
from .utils import convert_to_binary, carregar_dados

def treinar_modelo():
    df = carregar_dados("data/bootcamp_train.csv")
    X = df.drop(columns=["id", "classe"])
    y = df["classe"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline([
        ("preprocess", build_preprocessing_pipeline()),
        ("smote", SMOTE(random_state=42)),
        ("clf", XGBClassifier(eval_metric="mlogloss", random_state=42))
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.01, 0.1]
    }

    grid = GridSearchCV(pipeline, param_grid, scoring="f1_macro", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Melhores parâmetros:", grid.best_params_)
    print("Melhor F1-macro:", grid.best_score_)

    joblib.dump(grid.best_estimator_, "models/modelo_otimizado.pkl")

if __name__ == "__main__":
    treinar_modelo()