import numpy as np
import pandas as pd


def calculate_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    unique_values = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_values)
    value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(y_true)):
        true_idx = value_to_idx[y_true[i]]
        pred_idx = value_to_idx[y_pred[i]]
        conf_matrix[true_idx][pred_idx] += 1

    return conf_matrix

def calculate_accuracy_from_conf_matrix(conf_matrix):
    tn, fp = conf_matrix[0][0], conf_matrix[0][1]
    fn, tp = conf_matrix[1][0], conf_matrix[1][1]
    return (tp + tn) / (tp + tn + fp + fn)


def create_new_users(df: pd.DataFrame, n: int, id_col: str = "label") -> pd.DataFrame:
    """
    Selecciona n filas aleatorias de df y asigna a cada una un nuevo ID (entero)
    que no exista previamente en la columna id_col.

    Args:
        df (pd.DataFrame): DataFrame original que contiene la columna id_col.
        n (int): Número de filas a extraer y reasignar.
        id_col (str): Nombre de la columna de identificador de usuario.

    Returns:
        pd.DataFrame: Nuevo DataFrame con n filas (copia) y IDs renumerados.
    """
    # Copiar n muestras aleatorias
    new_data = df.sample(n, random_state=42).copy()

    # Calcular el máximo ID existente (considerando tanto df como las muestras)
    existing_ids = pd.concat([df[id_col], new_data[id_col]]).unique().astype(int)
    max_id = existing_ids.max()

    # Generar nuevos IDs consecutivos enteros
    new_ids = list(range(max_id + 1, max_id + 1 + n))
    new_data[id_col] = new_ids

    return new_data
