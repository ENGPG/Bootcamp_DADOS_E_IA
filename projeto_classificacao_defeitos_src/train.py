from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from .data_loader import load_and_prepare_data
from .model import create_model

def train_and_evaluate(filepath):
    X, y = load_and_prepare_data(filepath)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = create_model()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

    return pipeline
