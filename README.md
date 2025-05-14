# Projeto: Classificação de Defeitos em Chapas de Aço Inoxidável

Este repositório contém o código, modelos e materiais desenvolvidos para o projeto final do Bootcamp de Ciência de Dados e IA, cujo objetivo é construir um sistema inteligente para detectar e classificar defeitos em chapas de aço inoxidável com base em dados extraídos de imagens.

## Estrutura do Repositório

```
projeto-chapas-aco/
├── src/                     # Scripts principais do projeto
│   ├── preprocessing.py    # Funções de tratamento e preparação dos dados
│   ├── training.py         # Script de treino e validação dos modelos
│   ├── evaluation.py       # Avaliação e métricas dos modelos
│   └── utils.py            # Funções auxiliares (ex: conversão binária)
│
├── app/                    # API com FastAPI (extra)
│   └── main.py
│
├── dashboard/              # Dashboard Streamlit (extra)
│   └── app.py
│
├── models/                 # Modelos treinados e serializados
│   ├── modelo_otimizado.pkl
│   └── label_encoder.pkl
│
├── data/                   # Dados utilizados no projeto 
│   ├── bootcamp_train.csv
│   └── bootcamp_test.csv
│
├── notebooks/              # Análises e exploração inicial 
│   └── projeto_classificacao_defeitos__PAULO_GASPAR.ipynb
│
├── requirements.txt        # Dependências do projeto
├── README.md               # Este arquivo
└── Apresentacao.pptx       # Slides da apresentação
```

## Objetivo
Desenvolver um modelo de aprendizado de máquina capaz de identificar automaticamente o tipo de defeito presente em chapas de aço, com base em 31 variáveis extraídas das imagens (como área, perímetro, índices de luminosidade e geometria).

## Modelagem
- Algoritmos testados: RandomForest, XGBoost
- Técnicas usadas: SMOTE para balanceamento, GridSearchCV para tuning
- Métrica principal: F1-score macro

## Resultado Final
- Modelo final: **XGBoost**
- F1-macro (validação cruzada): **0.674**

## Extras
- [x] Exportação do modelo com `joblib`
- [x] Slides de apresentação
- [ ] API REST com FastAPI
- [ ] Dashboard interativo com Streamlit
- [ ] Dockerização (em progresso)

## Instalação
Clone o repositório e instale as dependências:
```bash
pip install -r requirements.txt
```

## Execução
Execute o script de treino com:
```bash
python src/training.py
```

Para rodar a API:
```bash
uvicorn app.main:app --reload
```

Para abrir o dashboard:
```bash
streamlit run dashboard/app.py
```

## Contato
Projeto desenvolvido por [PAULO FRANCISCO GASPAR].
Entre em contato: paulogasparginga2018@gmail.com

---
Bootcamp de Ciência de Dados e IA - SENAI | Maio/2025
