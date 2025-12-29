import os.path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# Configuración
TRAIN_PATH = os.path.join('dataset', 'train.csv')
TEST_PATH = os.path.join('dataset', 'testCleaned.csv')
OUTPUT_PATH = os.path.join('inference_outputs', 'TestPrediction_rf.csv')
LABEL_MAP = {'left': 0, 'moderate_left': 1, 'moderate_right': 2, 'right': 3}
inv_label_map = {v: k for k, v in LABEL_MAP.items()}

# Cargar datos
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# Procesamiento de features
def preprocess_data(df, label_encoder=None):
    # Eliminar columnas no usadas (incluyendo target)
    df = df.drop(
        columns=['id', 'tweet', 'ideology_binary', 'ideology_multiclass'],
        errors='ignore'
    )

    # Convertir 'label'
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])
    else:
        df['label'] = df['label'].apply(
            lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
        )

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['gender', 'profession'])

    return df, label_encoder


# Preparar datos
X_train, le = preprocess_data(train_df)
y_train = train_df['ideology_multiclass'].map(LABEL_MAP)  # Mapear desde el dataframe original

X_test, _ = preprocess_data(test_df, le)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Entrenar y predecir
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pd.DataFrame({
    'id': test_df['id'],
    'prediction': [inv_label_map[p] for p in rf.predict(X_test)]
}).to_csv(OUTPUT_PATH, index=False)

print(f"Predicciones RF guardadas en: {OUTPUT_PATH}")

# Calcular F1-score
true_df = pd.read_csv(OUTPUT_PATH)
logistic_df = pd.read_csv(os.path.join('inference_outputs', 'TestPrediction_logistic.csv'))
politibeto_df = pd.read_csv(os.path.join('inference_outputs', 'TestPrediction_politibeto.csv'))


def calc_f1(pred_df):
    merged = true_df.merge(pred_df, on='id')
    return f1_score(merged['prediction_x'], merged['prediction_y'], average='macro')


print(f"\nMacro F1 Logistic: {calc_f1(logistic_df):.4f}")
print(f"Macro F1 PolitiBeto: {calc_f1(politibeto_df):.4f}")