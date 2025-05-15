"""
Avaliação do modelo final com métricas e visualizações.
"""
import joblib
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def avaliar_modelo():
    df = pd.read_csv("data/bootcamp_train.csv")
    model = joblib.load("models/modelo_otimizado.pkl")
    X = df.drop(columns=["id", "classe"])
    y = df["classe"]

    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

    ConfusionMatrixDisplay.from_estimator(model, X, y)
    plt.title("Matriz de Confusão")
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    avaliar_modelo()