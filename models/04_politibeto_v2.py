import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

# Mapea etiquetas políticas a índices numéricos
LABEL_MAP = {
    'left': 0,
    'moderate_left': 1,
    'moderate_right': 2,
    'right': 3
}
# Parámetros de configuración del modelo y entrenamiento
MODEL_NAME = "nlp-cimat/politibeto"
MAX_LENGTH = 96
BATCH_SIZE = 24
LR = 2e-5
WEIGHT_DECAY = 0.01
EPSILON = 1e-8
EPOCHS = 5
WARMUP_STEPS = 300
SEEDS = [42, 56, 89]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join('model_politibeto', 'v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_tweet(text: str) -> str:
    """Normaliza menciones y URLs, y elimina espacios extra."""
    text = re.sub(r"@\w+", "[USER]", text)  # Sustituye menciones de usuario
    text = re.sub(r"http\S+", "[URL]", text)  # Sustituye URLs por marcador
    text = re.sub(r'\s+', ' ', text)              # Reduce múltiples espacios
    return text.strip()                            # Elimina espacios al inicio y fin

class PoliticalDataset(Dataset):
    """Dataset que aplica preprocesamiento y tokenización de tweets."""
    def __init__(self, texts, labels, tokenizer):
        self.tokenizer = tokenizer
        self.texts = [preprocess_tweet(t) for t in texts]  # Preprocesa cada texto
        self.labels = labels

    def __len__(self):
        return len(self.labels)  # Devuelve número de ejemplos

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        # Construye el diccionario de tensores para el modelo
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_single_model(seed, train_dataset, val_dataset, tokenizer):
    """Entrena un modelo con una semilla dada usando Trainer de Hugging Face."""
    torch.manual_seed(seed)          # Fija semilla PyTorch
    np.random.seed(seed)             # Fija semilla NumPy

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_MAP)
    ).to(DEVICE)                     # Carga y mueve modelo al dispositivo

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
        weight_decay=WEIGHT_DECAY, eps=EPSILON
    )                                  # Define optimizador AdamW
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=WARMUP_STEPS
    )                                  # Scheduler lineal para warmup

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds, average="macro")}

    # Configura directorio y argumentos de entrenamiento
    seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=seed_dir,
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=500,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        seed=seed
    )

    # Instancia el Trainer y entrena
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics
    )

    trainer.train()                  # Ejecuta el bucle de entrenamiento
    best_dir = os.path.join(OUTPUT_DIR, f"best_model_{seed}")
    os.makedirs(best_dir, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    return best_dir


def main():
    # Carga y mapeo de datos de train y development
    df_train = pd.read_csv(os.path.join('dataset', 'train.csv'))
    df_dev = pd.read_csv(os.path.join('dataset', 'development.csv'))

    for df in (df_train, df_dev):
        df['label'] = df['ideology_multiclass'].map(LABEL_MAP)  # Convierte etiquetas
        df.dropna(subset=['label'], inplace=True)              # Elimina filas sin etiqueta
        df['label'] = df['label'].astype(int)


    # División estratificada de train/validación
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_train['tweet'].tolist(),
        df_train['label'].tolist(),
        test_size=0.2,
        random_state=SEEDS[0],
        stratify=df_train['label']
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Carga tokenizador
    train_dataset = PoliticalDataset(train_texts, train_labels, tokenizer)
    val_dataset   = PoliticalDataset(val_texts, val_labels, tokenizer)

    # Entrena modelos con diferentes semillas y almacena rutas
    model_paths = []
    for seed in SEEDS:
        print(f"Entrenando seed {seed}")
        path = train_single_model(seed, train_dataset, val_dataset, tokenizer)
        model_paths.append(path)

    # Crea ensemble promediando logits de cada modelo en el set de dev
    models = [AutoModelForSequenceClassification.from_pretrained(p).to(DEVICE) for p in model_paths]
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    test_texts = df_dev['tweet'].tolist()
    test_labels = df_dev['label'].tolist()
    all_logits = []

    with torch.no_grad():
        for txt in test_texts:
            enc = tokenizer(
                preprocess_tweet(txt),
                truncation=True, padding='max_length',
                max_length=MAX_LENGTH, return_tensors='pt'
            ).to(DEVICE)
            # Obtiene logits de cada modelo y promedia
            logits = [m(**enc).logits.cpu().numpy() for m in models]
            all_logits.append(np.mean(logits, axis=0))

    preds = np.argmax(np.vstack(all_logits), axis=1)  # Predicciones finales
    # Muestra reporte de clasificación del ensemble
    print(classification_report(
        test_labels,
        preds,
        target_names=list(LABEL_MAP.keys()),
        zero_division=0
    ))

if __name__ == "__main__":
    main()

