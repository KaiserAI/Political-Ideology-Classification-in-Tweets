import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from utils import create_new_users

# Rutas y columna target
TRAIN_PATH = os.path.join("dataset", "train_wo_ideology_binary_numeric.csv")
TEST_PATH  = os.path.join("dataset", "test_wo_ideology_binary_numeric.csv")
TARGET_COLUMN = "ideology_multiclass"
FEATURE_ID    = "label"

# Configuración de modelos
MODEL_CONFIG = [
    (RandomForestClassifier, 'random_forest'),
    (XGBClassifier,        'xgboost_classifier'),
    (DecisionTreeClassifier,'decision_tree'),
    (KNeighborsClassifier, 'knn'),
    (LogisticRegression,   'logistic_regression'),
    (GaussianNB,           'naive_bayes'),
    (SVC,                  'svm'),
    ('RandomClassifier',   'random_baseline')
]

class RandomClassifier(BaseEstimator, ClassifierMixin):
    """Clasificador aleatorio multiclase"""
    def __init__(self, random_state=None):
        self.classes_ = None
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        rng = np.random.default_rng(self.random_state)
        return rng.choice(self.classes_, size=X.shape[0])

def load_and_preprocess_df(df: pd.DataFrame, scaler=None):
    """
    Dada un DataFrame, separa X/y, convierte FEATURE_ID a int y normaliza X.
    Devuelve (X_scaled, y, scaler).
    """
    # Separar target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values

    # Asegurar FEATURE_ID numérico
    X[FEATURE_ID] = X[FEATURE_ID].astype(int)

    # Escalado
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler

def load_and_preprocess(path: str, scaler=None):
    df = pd.read_csv(path)
    return load_and_preprocess_df(df, scaler)

def train_models(X_train, y_train):
    """Entrena todos los modelos y los devuelve en un dict."""
    models = {}
    for cls, name in MODEL_CONFIG:
        model = RandomClassifier() if name == 'random_baseline' else cls()
        model.fit(X_train, y_train)
        models[name] = model
    return models

def evaluate_models(models, X, y):
    """Evalúa cada modelo y devuelve un DataFrame con Accuracy y Macro‑F1."""
    results = {'Modelo': [], 'Accuracy': [], 'Macro-F1': []}
    for name, model in models.items():
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        f1 = report['macro avg']['f1-score']
        results['Modelo'].append(name)
        results['Accuracy'].append(round(acc, 4))
        results['Macro-F1'].append(round(f1, 4))
    return pd.DataFrame(results)

def plot_results(df, title):
    """Dibuja barras de Accuracy y Macro‑F1 para el DataFrame df."""
    plt.figure(figsize=(8,4))
    plt.bar(df['Modelo'], df['Accuracy'])
    plt.title(f"Accuracy - {title}")
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,4))
    plt.bar(df['Modelo'], df['Macro-F1'])
    plt.title(f"Macro-F1 - {title}")
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

def main():
    # 1) Cargar y preprocesar
    X_train, y_train, scaler = load_and_preprocess(TRAIN_PATH)
    X_test,  y_test,  _      = load_and_preprocess(TEST_PATH, scaler)

    # 2) Entrenar
    models = train_models(X_train, y_train)

    # 3) Evaluar en test original
    df_test = evaluate_models(models, X_test, y_test)
    print("\nResultados en Test original:")
    print(df_test.to_string(index=False))
    plot_results(df_test, 'Test Original con Label')

    # 4) Generar nuevos usuarios y evaluar
    df_full = pd.read_csv(TEST_PATH)
    new_data = create_new_users(df_full, n=100, id_col=FEATURE_ID)
    X_new, y_new, _ = load_and_preprocess_df(new_data, scaler)
    df_new = evaluate_models(models, X_new, y_new)
    print("\nResultados en Usuarios Nuevos:")
    print(df_new.to_string(index=False))
    plot_results(df_new, 'Usuarios Nuevos con Label')

if __name__ == "__main__":
    main()
