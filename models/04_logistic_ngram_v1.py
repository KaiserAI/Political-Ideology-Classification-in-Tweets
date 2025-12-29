import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

# Función placeholder para calcular métricas de legibilidad (9 características)
def readability_features(texts):
    return np.zeros((len(texts), 9))  # Devuelve ceros si no se implementa

# Rutas de archivos de entrenamiento y prueba
TRAIN_PATH = os.path.join('dataset', 'train.csv')
TEST_PATH  = os.path.join('dataset', 'development.csv')

# Carga de datos desde CSV
df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

# Preprocesamiento básico y mapeo de etiquetas a valores numéricos
X_train = df_train['tweet'].astype(str)
y_train = df_train['ideology_multiclass'].map({'left':0, 'moderate_left':1, 'moderate_right':2, 'right':3})

X_test  = df_test['tweet'].astype(str)
y_test  = df_test['ideology_multiclass'].map({'left':0, 'moderate_left':1, 'moderate_right':2, 'right':3})

# Definición de vectores TF-IDF para palabras y caracteres
tfidf_word = TfidfVectorizer(ngram_range=(1,4), analyzer='word', max_features=5000)
tfidf_char = TfidfVectorizer(ngram_range=(1,1), analyzer='char')

# Pipeline para características de legibilidad usando FunctionTransformer
readability_pipe = Pipeline([
    ('read', FunctionTransformer(lambda texts: readability_features(texts), validate=False))
])

# Unión de todas las características en un solo transformador
features = FeatureUnion([
    ('words', tfidf_word),             # N‑gramas de palabras
    ('chars', tfidf_char),             # Caracteres individuales
    ('readability', readability_pipe)  # Métricas de legibilidad
])

# Pipeline completo con extracción de características y regresión logística
pipe = Pipeline([
    ('features', features),
    ('clf', LogisticRegression(penalty='l2', solver='liblinear', multi_class='auto'))
])

# Definición de la cuadrícula de búsqueda para el hiperparámetro C
param_grid = {'clf__C': [0.1, 1, 10]}

# Configuración de GridSearchCV con validación cruzada y scoring macro‑F1
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

# Entrenamiento del modelo con búsqueda de hiperparámetros
def main():
    print("Entrenando modelo...")
    grid.fit(X_train, y_train)           # Ajuste de GridSearchCV

    best_C = grid.best_params_['clf__C']  # Mejor valor de C encontrado
    print(f"\nMejor parámetro C: {best_C}")

    # Predicción y evaluación en el conjunto de prueba
    preds = grid.predict(X_test)
    report = classification_report(
        y_test,
        preds,
        target_names=['left','moderate_left','moderate_right','right'],
        digits=4
    )
    macro_f1 = f1_score(y_test, preds, average='macro')

    print("\n == Logistic Regression con N-gramas ==")
    print(report)
    print(f"Macro F1 score: {macro_f1:.4f}")

if __name__ == '__main__':
    main()
