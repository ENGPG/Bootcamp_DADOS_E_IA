y_pred = final_model.predict(X_val)

ConfusionMatrixDisplay.from_estimator(
    final_model, X_val, y_val,
    display_labels=le.classes_,
    cmap='Blues'
)
plt.title("Matriz de Confusão - Validação")
plt.xticks(rotation=45)
plt.grid(False)
plt.tight_layout()
plt.show()

importances = final_model.named_steps['clf'].feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Importância das Features no Modelo Final")
plt.xlabel("Importância")
plt.tight_layout()
plt.grid(True)
plt.show()

X_transformed = preprocess_pipeline.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_transformed)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=le.inverse_transform(y), palette='Set2')
plt.title("Visualização PCA (2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

f1_rf = cross_val_score(best_rf, X, y, scoring='f1_macro', cv=5)
f1_xgb = cross_val_score(best_xgb, X, y, scoring='f1_macro', cv=5)

print("F1-macro RF:", f1_rf)
print("Média F1 RF:", f1_rf.mean())
print("F1-macro XGB:", f1_xgb)
print("Média F1 XGB:", f1_xgb.mean())

train_sizes, train_scores, val_scores = learning_curve(
    final_model, X, y, cv=5, scoring='f1_macro',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, 'o-', label='F1 Treinamento')
plt.plot(train_sizes, val_mean, 'o-', label='F1 Validação')
plt.title("Curva de Aprendizado")
plt.xlabel("Tamanho do Conjunto de Treinamento")
plt.ylabel("F1 Macro")
plt.legend()
plt.grid(True)
plt.show()
