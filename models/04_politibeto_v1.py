import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Mapeo de etiquetas políticas a valores numéricos
LABEL_MAP = {
    'left': 0,
    'moderate_left': 1,
    'moderate_right': 2,
    'right': 3
}

# Configuración de parámetros y rutas de archivos
MODEL_NAME = "nlp-cimat/politibeto"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join('model_politibeto', 'v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_tweet(text):
    """
    Normaliza menciones y URLs, elimina espacios extra,
    reduce repeticiones de caracteres y conserva solo ASCII.
    """
    text = re.sub(r"@\w+", "[USER]", text)  # Reemplaza menciones de usuario
    text = re.sub(r"http\S+", "[URL]", text)  # Reemplaza URLs
    text = re.sub(r'\s+', ' ', text)              # Elimina espacios múltiples
    text = re.sub(r'(.)\1{2,}', r'\1', text)     # Reduce repeticiones largas
    text = text.encode('ascii', 'ignore').decode('ascii')  # Filtra caracteres no ASCII
    return text.strip()

class PoliticalDataset(Dataset):
    """
    Dataset que aplica preprocesamiento y tokenización para cada tweet.
    """
    def __init__(self, texts, labels, tokenizer):
        self.texts = [preprocess_tweet(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),  # Tensor de IDs de tokens
            'attention_mask': enc['attention_mask'].squeeze(),  # Máscara de atención
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  # Etiqueta como tensor
        }

def load_data(train_path, test_path):
    """
    Carga los CSV de entrenamiento y test, y mapea las etiquetas.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for df in (train_df, test_df):
        df['label'] = df['ideology_multiclass'].map(LABEL_MAP)  # Convierte a numerico
        df.dropna(subset=['label'], inplace=True)  # Elimina filas sin etiqueta, que no hay, pero por si acaso en otro data set
        df['label'] = df['label'].astype(int)

    return train_df, test_df

def initialize_model():
    """
    Carga el modelo base y descongela las últimas 4 capas del encoder y el clasificador.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP)
    )
    # Descongela últimas 4 capas del encoder BERT
    for param in model.bert.encoder.layer[-4:].parameters():
        param.requires_grad = True
    # Descongela parámetros del clasificador
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model.to(DEVICE)

def train(model, train_loader, val_loader, class_weights):
    """
    Entrena el modelo usando AdamW, scheduler lineal y early stopping.
    """
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * EPOCHS
    )
    loss_fn = CrossEntropyLoss(weight=class_weights.to(DEVICE))  # Pérdida con ponderación de clases

    best_f1 = 0
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f'Época {epoch+1}'):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}  # Mueve inputs a GPU
            labels = batch['labels'].to(DEVICE)
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Grad clip
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        val_f1 = evaluate(model, val_loader)
        print(f"F1 validación: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)  # Guarda mejor modelo
        else:
            print("Detención temprana")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))  # Carga modelo óptimo
    return model

def evaluate(model, loader):
    """
    Calcula el puntaje F1 macro en un DataLoader dado.
    """
    model.eval()
    preds, labels_true = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].cpu().numpy()
            outputs = model(**inputs)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels_true.extend(labels)
    return f1_score(labels_true, preds, average='macro')

def main():
    """
    Orquesta la carga de datos, entrenamiento, evaluación final y guardado.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_df, test_df = load_data(
        os.path.join('dataset', 'train.csv'),
        os.path.join('dataset', 'development.csv')
    )
    # División estratificada de train/validación
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['tweet'].tolist(),
        train_df['label'].tolist(),
        test_size=0.1,
        random_state=SEED,
        stratify=train_df['label']
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = PoliticalDataset(train_texts, train_labels, tokenizer)
    val_ds = PoliticalDataset(val_texts, val_labels, tokenizer)
    test_ds = PoliticalDataset(test_df['tweet'].tolist(), test_df['label'].tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Cálculo de pesos para clases desequilibradas
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    model = initialize_model()  # Inicializa y descongela capas
    model = train(model, train_loader, val_loader, class_weights)

    f1_test = evaluate(model, test_loader)
    print(f"Macro F1 en test: {f1_test:.4f}")

    # Guardado final del modelo y tokenizer
    model.save_pretrained(os.path.join(OUTPUT_DIR))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR))

    # Reporte de clasificación detallado en test
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].cpu().numpy()
            outputs = model(**inputs)
            y_pred.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            y_true.extend(labels)

    print(classification_report(
        y_true,
        y_pred,
        target_names=list(LABEL_MAP.keys()),
        labels=list(LABEL_MAP.values()),
        zero_division=0
    ))

if __name__ == "__main__":
    main()