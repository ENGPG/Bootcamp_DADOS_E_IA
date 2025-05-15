df = pd.read_csv("bootcamp_train.csv")

binary_cols = ['tipo_do_aço_A300', 'tipo_do_aço_A400']
falhas = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']

for col in binary_cols + falhas:
    df[col] = df[col].apply(convert_to_binary)

df[binary_cols + falhas] = SimpleImputer(strategy='most_frequent').fit_transform(df[binary_cols + falhas])

df = df[df[falhas].sum(axis=1) == 1].copy()
df['classe'] = df[falhas].idxmax(axis=1)
df.drop(columns=falhas, inplace=True)

le = LabelEncoder()
df['classe'] = le.fit_transform(df['classe'])

X = df.drop(columns=['id', 'classe'])
y = df['classe']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocess_pipeline = SklearnPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
