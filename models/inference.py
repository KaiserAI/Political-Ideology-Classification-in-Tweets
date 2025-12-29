import numpy as np

# Parche obligatorio para numpy.dtypes
class NumpyDTypesMock:
    UInt32DType = np.dtype('uint32')
    UInt64DType = np.dtype('uint64')

if not hasattr(np, 'dtypes'):
    np.dtypes = NumpyDTypesMock()
else:
    np.dtypes.UInt32DType = np.dtype('uint32')
    np.dtypes.UInt64DType = np.dtype('uint64')

import re
import os
import joblib
import pandas as pd
import torch
from pathlib import Path
from safetensors import safe_open
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).parent.absolute()

TEST_PATH = BASE_DIR / 'dataset' / 'testCleaned.csv'
LOGISTIC_PATH = BASE_DIR / 'models' / 'ideology_pipeline.joblib'
POLITI_BASE = BASE_DIR / 'model_politibeto' / 'v2'
SEEDS = [42, 56, 89]
MAX_LENGTH = 96
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = BASE_DIR / 'inference_outputs'

LABEL_MAP = {'left': 0, 'moderate_left': 1, 'moderate_right': 2, 'right': 3}
inv_label_map = {v: k for k, v in LABEL_MAP.items()}  # Mapeo inverso

def parse_text_standard(txt):
    nums = re.findall(r"(\d+)", txt)
    return sum(map(int, nums)) / len(nums) if nums else np.nan

def compute_readability(texts):
    import textstat
    feats = []
    for t in texts:
        try:
            std_val = parse_text_standard(textstat.text_standard(t))
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
    arr[np.isnan(arr)] = np.take(col_means, np.where(np.isnan(arr))[1])
    return arr

def stylistic_features(texts):
    feats = []
    emoji_re = re.compile(r"[\U0001F600-\U0001F64F]")
    for t in texts:
        tokens = t.split()
        n_tokens = len(tokens)
        feats.append([
            n_tokens,
            len(t),
            t.count('.') + t.count('!') + t.count('?'),
            sum(w.isupper() for w in tokens) / max(n_tokens, 1),
            t.count('!'),
            t.count('?'),
            t.count('#'),
            t.count('@'),
            len(emoji_re.findall(t)),
        ])
    return np.array(feats, dtype=float)

# =============================================
print("Inferencia con Logistic N-gram...")
df_test = pd.read_csv(TEST_PATH)
pipeline_log = joblib.load(LOGISTIC_PATH)
pred_log = pipeline_log.predict(df_test['tweet'].astype(str))

# Convertir a etiquetas
pred_log_labels = [inv_label_map[p] for p in pred_log]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'id': df_test['id'], 'prediction': pred_log_labels}) \
    .to_csv(OUTPUT_DIR / 'TestPrediction_logistic.csv', index=False)
print(f"Guardado: {OUTPUT_DIR / 'TestPrediction_logistic.csv'}")


# =============================================
print("\nInferencia con PolitiBeto...")

# 1. Carga del tokenizador
tokenizer_dir = POLITI_BASE / f"best_model_{SEEDS[0]}"
assert tokenizer_dir.is_dir(), f"Directorio tokenizador: {tokenizer_dir} no existe"

tokenizer = AutoTokenizer.from_pretrained(
    str(tokenizer_dir),
    local_files_only=True
)

# 2. Carga de modelos
models = []
for seed in SEEDS:
    model_dir = POLITI_BASE / f"best_model_{seed}"
    model_path = model_dir / "model.safetensors"

    assert model_path.exists(), f"Falta model.safetensors en {model_dir}"

    # Cargar configuración directamente desde archivo
    with open(model_dir / "config.json") as f:
        config = AutoConfig.from_pretrained(str(model_dir))

    # Carga segura de pesos
    with safe_open(str(model_path), framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    # Construir modelo manualmente
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    models.append(model)

# 3. Pipeline de inferencia
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

loader = DataLoader(
    TextDataset(df_test['tweet'].astype(str).tolist(), tokenizer),
    batch_size=32,
    shuffle=False,
    num_workers=4
)

all_logits = []
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch_logits = [model(**batch).logits.cpu().numpy() for model in models]
        all_logits.append(np.mean(batch_logits, axis=0))

pred_poli = np.argmax(np.concatenate(all_logits), axis=1)

# Convertir a etiquetas
pred_poli_labels = [inv_label_map[p] for p in pred_poli]

# 4. Guardar resultados
pd.DataFrame({'id': df_test['id'], 'prediction': pred_poli_labels}) \
    .to_csv(OUTPUT_DIR / 'TestPrediction_politibeto.csv', index=False)
print(f"Guardado: {OUTPUT_DIR / 'TestPrediction_politibeto.csv'}")
