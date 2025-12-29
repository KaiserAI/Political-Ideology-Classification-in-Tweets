import os
import pandas as pd
import torch
import torch.nn as nn
# Añadir fix de tipos antes de transformers
import numpy as np
import numpy.dtypes as dtypes
if not hasattr(dtypes, 'UInt32DType'):
    dtypes.UInt32DType = np.dtype('uint32')
if not hasattr(dtypes, 'UInt64DType'):
    dtypes.UInt64DType = np.dtype('uint64')

# Ahora importar transformers
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score

# Rutas y parámetros principales
TRAIN_PATH   = os.path.join('dataset', 'train.csv')
TEST_PATH    = os.path.join('dataset', 'development.csv')
RANDOM_SEED  = 42
MODEL_BETO   = 'dccuchile/bert-base-spanish-wwm-cased'
MODEL_MARIA  = 'PlanTL-GOB-ES/roberta-large-bne'
BATCH_SIZE   = 8
LR           = 3e-5
EPOCHS       = 1
MAX_LEN      = 512
OUTPUT_DIR   = os.path.join('model_dual_bert', 'v1')

# Mapeo de las etiquetas de ideología
LABEL_MAP = {
    'left': 0,
    'moderate_left': 1,
    'moderate_right': 2,
    'right': 3
}

class TweetDataset(Dataset):
    # Tokeniza textos y devuelve tensores para entrenamiento/evaluación
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class DualBertClassifier(nn.Module):
    # Modelo dual que concatena representaciones de Beto y María
    def __init__(self):
        super().__init__()
        self.beto  = AutoModel.from_pretrained(MODEL_BETO)
        self.maria = AutoModel.from_pretrained(MODEL_MARIA)
        hidden1 = self.beto.config.hidden_size
        hidden2 = self.maria.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(hidden1 + hidden2, len(LABEL_MAP))

    def forward(self, input_ids, attention_mask):
        out1 = self.beto(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        out2 = self.maria(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        x = torch.cat([out1, out2], dim=1)
        x = self.dropout(x)
        return self.classifier(x)


def train_and_evaluate():
    # Cargar datos y preparar conjuntos de train y test
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    train_texts  = train_df['tweet'].astype(str).tolist()
    train_labels = train_df['ideology_multiclass'].map(LABEL_MAP).tolist()
    test_texts   = test_df['tweet'].astype(str).tolist()
    test_labels  = test_df['ideology_multiclass'].map(LABEL_MAP).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BETO)
    train_ds  = TweetDataset(train_texts, train_labels, tokenizer)
    test_ds   = TweetDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualBertClassifier().to(device)

    # Congelar Beto y María para entrenar solo el clasificador
    for param in model.beto.parameters(): param.requires_grad = False
    for param in model.maria.parameters(): param.requires_grad = False

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    # Fase de entrenamiento
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            input_ids     = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels        = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}")

    # Evaluación en development set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids     = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            preds  = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)

    print("\nClassification Report:")
    print(classification_report(test_labels, all_preds, target_names=list(LABEL_MAP.keys()), digits=4))
    print(f"Macro F1 score: {f1_score(test_labels, all_preds, average='macro'):.4f}")

    # Guardado de modelo y tokenizador
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_dual_bert_v1.bin'))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modelo y tokenizador guardados en {OUTPUT_DIR}")

if __name__ == '__main__':
    train_and_evaluate()
