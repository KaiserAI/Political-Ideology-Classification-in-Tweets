import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

TRAIN_PATH   = os.path.join('dataset', 'train.csv')
DEV_PATH     = os.path.join('dataset', 'development.csv')
SEED         = 42
MODEL_BETO   = 'dccuchile/bert-base-spanish-wwm-cased'
MODEL_MARIA  = 'PlanTL-GOB-ES/roberta-large-bne'
MAX_LEN      = 128
BATCH_SIZE   = 16
ACCUM_STEPS  = 2
LR           = 3e-5
EPOCHS       = 1
FREEZE_RATIO = 0.5
OUTPUT_DIR   = os.path.join('model_dual_bert', 'v2')
NUM_WORKERS  = 4

LABEL_MAP = {'left':0,'moderate_left':1,'moderate_right':2,'right':3}

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tokenizer = texts, labels, tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        return {
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class DualBertWithLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        # Carga de las dos arquitecturas base (Beto y María)
        base_beto  = AutoModel.from_pretrained(MODEL_BETO)
        base_maria = AutoModel.from_pretrained(MODEL_MARIA)
        # Configuración de LoRA para reducir parámetros entrenables
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05
        )
        # Aplicación de LoRA a ambos modelos
        self.beto  = get_peft_model(base_beto,  peft_config)
        self.maria = get_peft_model(base_maria, peft_config)
        # Activar checkpointing para ahorrar memoria en GPU
        self.beto.gradient_checkpointing_enable()
        self.maria.gradient_checkpointing_enable()
        # Definición de la capa de normalización y clasificador final
        h1, h2 = self.beto.config.hidden_size, self.maria.config.hidden_size
        self.norm = nn.LayerNorm(h1 + h2)
        self.mlp  = nn.Sequential(
            nn.Linear(h1 + h2, 512), nn.ReLU(), nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(512, len(LABEL_MAP))
        # Congelar capas inferiores para reducir overfitting y coste computacional
        self._freeze_lower_layers()

    def _freeze_lower_layers(self):
        for model in (self.beto, self.maria):
            layers = model.base_model.encoder.layer
            freeze_cnt = int(FREEZE_RATIO * len(layers))
            # Solo entrenamos la mitad superior de las capas
            for layer in layers[:freeze_cnt]:
                for p in layer.parameters(): p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Obtener la representación [CLS] de ambos modelos
        o1 = self.beto.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        o2 = self.maria.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        # Concatenar, normalizar, pasar por MLP y clasificar
        x = torch.cat((o1, o2), dim=1)
        x = self.norm(x)
        x = self.mlp(x)
        return self.classifier(x)


def train_and_evaluate():
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Lectura de datos y mapeo de etiquetas
    train_df = pd.read_csv(TRAIN_PATH)
    dev_df   = pd.read_csv(DEV_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BETO)
    X_train = train_df['tweet'].astype(str).tolist()
    y_train = train_df['ideology_multiclass'].map(LABEL_MAP).tolist()
    X_dev   = dev_df['tweet'].astype(str).tolist()
    y_dev   = dev_df['ideology_multiclass'].map(LABEL_MAP).tolist()
    # Preparación de DataLoaders
    train_loader = DataLoader(TweetDataset(X_train, y_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dev_loader   = DataLoader(TweetDataset(X_dev,   y_dev,   tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    # Instanciación de modelo, optimizador y funciones de pérdida
    model = DualBertWithLoRA().to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    scaler  = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()
        # Entrenamiento con acumulación de gradientes
        for step, batch in enumerate(train_loader, 1):
            ids = batch['input_ids'].to(device)
            m   = batch['attention_mask'].to(device)
            lbl = batch['labels'].to(device)
            with torch.cuda.amp.autocast():
                logits = model(ids, m)
                loss   = loss_fn(logits, lbl) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if step % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * ACCUM_STEPS
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}")
    # Evaluación en el conjunto de validación
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dev_loader:
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            pred = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(pred); trues.extend(batch['labels'].tolist())
    # Métricas de desempeño
    print("Dev Macro-F1:", f1_score(trues, preds, average='macro'))
    print(classification_report(trues, preds, target_names=LABEL_MAP.keys()))
    # Guardado de modelo y tokenizador
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Modelo guardado en", OUTPUT_DIR)

if __name__ == '__main__':
    train_and_evaluate()
