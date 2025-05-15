from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__class_weight': ['balanced']
}

grid_rf = GridSearchCV(pipeline, param_grid_rf, scoring='f1_macro', cv=5, verbose=1, n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("Melhores parâmetros RF:", grid_rf.best_params_)
print("Melhor F1 RF:", grid_rf.best_score_)
best_rf = grid_rf.best_estimator_

pipeline.set_params(clf=XGBClassifier(eval_metric='mlogloss', random_state=42))

param_grid_xgb = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 6],
    'clf__learning_rate': [0.01, 0.1]
}

grid_xgb = GridSearchCV(pipeline, param_grid_xgb, scoring='f1_macro', cv=5, verbose=1, n_jobs=-1)
grid_xgb.fit(X_train, y_train)

print("Melhores parâmetros XGB:", grid_xgb.best_params_)
print("Melhor F1 XGB:", grid_xgb.best_score_)
best_xgb = grid_xgb.best_estimator_

final_model = best_xgb if grid_xgb.best_score_ > grid_rf.best_score_ else best_rf
final_model.fit(X_train, y_train)

joblib.dump(final_model, 'modelo_otimizado.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Modelo final salvo com sucesso!")
