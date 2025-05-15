from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from .preprocessing import create_preprocessing_pipeline

def create_model():
    preprocess = create_preprocessing_pipeline()

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )

    pipeline = ImbPipeline([
        ('preprocess', preprocess),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    return pipeline
