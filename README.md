# Projeto Final - Classificação de Defeitos em Chapas de Aço

## Estrutura

```
📁 data/         -> Dados originais
📁 notebooks/    -> Análise e modelagem
📁 src/          -> Scripts: pré-processamento, treino, API
📁 models/       -> Modelos treinados (.joblib)
```

## Como usar

1. Edite e execute `src/train_model.py` para treinar o modelo.
2. Use `src/predict.py` para fazer previsões.
3. Rode a API com:

```bash
uvicorn src.api:app --reload
```

## Requisitos

- Python 3.9+
- scikit-learn
- pandas
- joblib
- fastapi
- uvicorn
