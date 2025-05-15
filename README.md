# Projeto Final - Classificação de Defeitos em Chapas de Aço

## Estrutura

```
📁 Slide/         -> arquivo de Apresentacao_Projeto_Chapas_Aco.pdf
📁 data/         -> Dados originais (Bootcamp_train.csv, Bootcamp_test.csv)
📁 notebooks/    -> Análise exploratória e modelagem (Jupyter Notebooks)
📁 src/          -> Scripts principais: pré-processamento, treino, avaliação
📁 models/       -> Modelos treinados (.joblib)


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
