# Projeto Final - Classificação de Defeitos em Chapas de Aço

## Estrutura

```
📁 data/         -> Dados originais (Bootcamp_train.csv, Bootcamp_test.csv)
📁 notebooks/    -> Análise exploratória e modelagem (Jupyter Notebooks)
📁 src/          -> Scripts principais: pré-processamento, treino, avaliação
📁 models/       -> Modelos treinados (.joblib)
📁 api/          -> Interface de inferência com FastAPI (opcional)


```


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
