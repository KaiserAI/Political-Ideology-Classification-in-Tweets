import os
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

# Función para extraer medias de números en texto
def parse_text_standard(txt):
    nums = re.findall(r"(\d+)", txt)
    if nums:
        vals = list(map(int, nums))
        return sum(vals) / len(vals)
    else:
        return np.nan

# Calcula nueve métricas de legibilidad y rellena NaN con medias de columna
def compute_readability(texts):
    import textstat  # Librería para métricas de legibilidad
    feats = []
    for t in texts:
        try:
            std = textstat.text_standard(t)
            std_val = parse_text_standard(std)
        except:
            std_val = np.nan
        feats.append([
            textstat.flesch_reading_ease(t),
            textstat.flesch_kincaid_grade(t),
            textstat.coleman_liau_index(t),
            textstat.gunning_fog(t),
            textstat.smog_index(t),
            textstat.automated_readability_index(t),
            textstat.dale_chall_readability_score(t),
            textstat.linsear_write_formula(t),
            std_val
        ])
    arr = np.array(feats, dtype=float)
    col_means = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_means, inds[1])
    return arr

# Extrae características estilísticas: tokens, longitud, puntuación, emojis, mayúsculas
def stylistic_features(texts):
    feats = []
    emoji_re = re.compile(r"[\U0001F600-\U0001F64F]")
    for t in texts:
        tokens = t.split()
        n_tokens = len(tokens)
        n_chars = len(t)
        n_sent = t.count('.') + t.count('!') + t.count('?')
        pct_upper = sum(w.isupper() for w in tokens) / max(n_tokens,1)
        feats.append([
            n_tokens,
            n_chars,
            n_sent,
            pct_upper,
            t.count('!'),
            t.count('?'),
            t.count('#'),
            t.count('@'),
            len(emoji_re.findall(t)),
        ])
    return np.array(feats, dtype=float)

# Carga de datos de entrenamiento y desarrollo
df_train = pd.read_csv(os.path.join('dataset', 'train.csv'))
df_dev   = pd.read_csv(os.path.join('dataset', 'development.csv'))

# Extracción de textos y mapeo de etiquetas a enteros
X_train = df_train['tweet'].astype(str)
y_train = df_train['ideology_multiclass'].map({
    'left': 0,
    'moderate_left': 1,
    'moderate_right': 2,
    'right': 3
})
X_dev   = df_dev['tweet'].astype(str)
y_dev   = df_dev['ideology_multiclass'].map({
    'left': 0,
    'moderate_left': 1,
    'moderate_right': 2,
    'right': 3
})

# Pipeline para métricas de legibilidad y escalado
read_pipe = Pipeline([
    ('calc', FunctionTransformer(compute_readability, validate=False)),
    ('scale', StandardScaler())
])

# Pipeline para características estilísticas y escalado
styl_pipe = Pipeline([
    ('calc', FunctionTransformer(stylistic_features, validate=False)),
    ('scale', StandardScaler())
])

# TF-IDF de palabras y caracteres con parámetros ajustados
word_tfidf = TfidfVectorizer(ngram_range=(1,4), max_features=15000,
                             min_df=5, max_df=0.8)
char_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1,3),
                             min_df=5, max_df=0.8)

# Unión de todas las fuentes de características en un solo transformador
features = FeatureUnion([
    ('word', word_tfidf),    # N-gramas de palabras
    ('char', char_tfidf),    # N-gramas de caracteres
    ('read', read_pipe),      # Métricas de legibilidad
    ('styl', styl_pipe),      # Características estilísticas
])

# Definición de clasificadores base con ponderación de clases
lr = LogisticRegression(penalty='l2', class_weight='balanced', solver='saga', max_iter=5000)
svm = CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=5000), cv=3, method='sigmoid')
rf  = RandomForestClassifier(class_weight='balanced', n_estimators=200)

# Selector de características basado en L1 para reducir dimensionalidad
selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='saga', class_weight='balanced', max_iter=5000),
    threshold='median'
)

# Ensemble por votación blanda de los tres modelos base
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
    voting='soft',
    n_jobs=-1
)

# Pipeline final que combina extracción, selección y clasificación
pipeline = Pipeline([
    ('features', features),
    ('select', selector),
    ('clf', ensemble)
])

# Definición de la cuadrícula para Grid Search de hiperparámetros
param_grid = {
    'clf__lr__C': [0.1, 1],                   # C para regresión logística
    'clf__svm__base_estimator__C': [0.1, 1],  # C para SVM calibrada
    'clf__rf__n_estimators': [100, 200],      # Número de árboles en RF
}

# Configura GridSearchCV con validación cruzada y scoring macro-F1
grid = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=3, n_jobs=-1, verbose=2, error_score='raise')

# Entrenamiento con búsqueda de hiperparámetros
print("=== Training con GridSearchCV (cv=3) ===")
grid.fit(X_train, y_train)

# Obtiene el mejor modelo tras Grid Search
best_model = grid.best_estimator_

# Evaluación del modelo en el conjunto de desarrollo
y_dev_pred = best_model.predict(X_dev)
print("\n=== Evaluación Dev Set ===")
print(classification_report(y_dev, y_dev_pred, target_names=['left','moderate_left','moderate_right','right'], digits=4))
print(f"Dev F1-macro: {f1_score(y_dev, y_dev_pred, average='macro'):.4f}")

# Guarda el modelo final en formato joblib
os.makedirs('model_logistic_ngram', exist_ok=True)
model_path = os.path.join('model_logistic_ngram', 'ideology_pipeline.joblib')
joblib.dump(best_model, model_path)
print(f"Modelo guardado en: {model_path}")
