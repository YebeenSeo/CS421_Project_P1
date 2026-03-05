import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score


# Data loading
def load_csv(path):
    return pd.read_csv(path, engine="python", on_bad_lines="skip")

train = load_csv("data/trac2_CONVT_train.csv")
dev   = load_csv("data/trac2_CONVT_dev.csv")
test  = load_csv("data/trac2_CONVT_test.csv")

# dev column mismatch fix (use intersection)
common_cols = [c for c in train.columns if c in dev.columns]
dev = dev[common_cols]

# polarity cleanup: merge rare class 3 into 2
train["EmotionalPolarity"] = train["EmotionalPolarity"].astype(int)
dev["EmotionalPolarity"]   = dev["EmotionalPolarity"].astype(int)
train.loc[train["EmotionalPolarity"] == 3, "EmotionalPolarity"] = 2
dev.loc[dev["EmotionalPolarity"] == 3, "EmotionalPolarity"]     = 2

# Make sure required columns exist
REQ_TRAIN = ["id", "text", "Emotion", "Empathy", "EmotionalPolarity"]
for c in REQ_TRAIN:
    if c not in train.columns:
        raise ValueError(f"Missing column in train: {c}")

# add simple conversation context (put just in case)
def build_context(df, n_prev=0):
    """
    If n_prev=0: use just current text.
    If n_prev>0: prepend up to n_prev previous turns within same conversation_id.
    """
    if n_prev <= 0 or "conversation_id" not in df.columns or "turn_id" not in df.columns:
        return df["text"].fillna("").astype(str).tolist()

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df.sort_values(["conversation_id", "turn_id", "id"])

    contexts = []
    buffer = []
    last_cid = None

    for _, row in df.iterrows():
        cid = row["conversation_id"]
        if last_cid != cid:
            buffer = []
            last_cid = cid

        prev = buffer[-n_prev:] if n_prev > 0 else []
        ctx = " ".join(prev + [row["text"]]).strip()
        contexts.append(ctx)

        buffer.append(row["text"])

    return contexts

# Start simple: no context
train_texts = build_context(train, n_prev=0)
dev_texts   = build_context(dev, n_prev=0)
test_texts  = build_context(test, n_prev=0)


# Embeddings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"  # fast + good baseline
embedder = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)

def encode_texts(texts, batch_size=64):
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        emb = embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        vecs.append(emb)
    return np.vstack(vecs)

X_train = encode_texts(train_texts)
X_dev   = encode_texts(dev_texts)
X_test  = encode_texts(test_texts)

y_emotion_train = train["Emotion"].astype(float).to_numpy()
y_empathy_train = train["Empathy"].astype(float).to_numpy()
y_polar_train   = train["EmotionalPolarity"].astype(int).to_numpy()

y_emotion_dev = dev["Emotion"].astype(float).to_numpy()
y_empathy_dev = dev["Empathy"].astype(float).to_numpy()
y_polar_dev   = dev["EmotionalPolarity"].astype(int).to_numpy()


# Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, X, y_emotion=None, y_empathy=None, y_polar=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_emotion = None if y_emotion is None else torch.tensor(y_emotion, dtype=torch.float32)
        self.y_empathy = None if y_empathy is None else torch.tensor(y_empathy, dtype=torch.float32)
        self.y_polar   = None if y_polar   is None else torch.tensor(y_polar,   dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        item = {"x": self.X[idx]}
        if self.y_emotion is not None:
            item["emotion"] = self.y_emotion[idx]
            item["empathy"] = self.y_empathy[idx]
            item["polarity"] = self.y_polar[idx]
        return item


# Model
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.2, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_emotion = nn.Linear(hidden_dim, 1)
        self.head_empathy = nn.Linear(hidden_dim, 1)
        self.head_polar   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.backbone(x)
        emotion = self.head_emotion(h).squeeze(-1)
        empathy = self.head_empathy(h).squeeze(-1)
        polar_logits = self.head_polar(h)
        return emotion, empathy, polar_logits


# Train
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

train_ds = MultiTaskDataset(X_train, y_emotion_train, y_empathy_train, y_polar_train)
dev_ds   = MultiTaskDataset(X_dev,   y_emotion_dev,   y_empathy_dev,   y_polar_dev)
test_ds  = MultiTaskDataset(X_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

model = MLP(in_dim=X_train.shape[1]).to("cpu")  # Q1 says CPU training
opt = torch.optim.Adam(model.parameters(), lr=LR)

loss_reg = nn.MSELoss()
loss_cls = nn.CrossEntropyLoss()

def evaluate():
    model.eval()
    emo_preds, emp_preds, pol_preds = [], [], []
    emo_gold, emp_gold, pol_gold = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            x = batch["x"]
            emotion, empathy, pol_logits = model(x)
            emo_preds.extend(emotion.numpy().tolist())
            emp_preds.extend(empathy.numpy().tolist())
            pol_preds.extend(torch.argmax(pol_logits, dim=1).numpy().tolist())

            emo_gold.extend(batch["emotion"].numpy().tolist())
            emp_gold.extend(batch["empathy"].numpy().tolist())
            pol_gold.extend(batch["polarity"].numpy().tolist())

    emo_mae = mean_absolute_error(emo_gold, emo_preds)
    emp_mae = mean_absolute_error(emp_gold, emp_preds)
    acc = accuracy_score(pol_gold, pol_preds)
    report = classification_report(pol_gold, pol_preds, digits=4, zero_division=0)
    return emo_mae, emp_mae, acc, report

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        x = batch["x"]
        emotion, empathy, pol_logits = model(x)

        l1 = loss_reg(emotion, batch["emotion"])
        l2 = loss_reg(empathy, batch["empathy"])
        l3 = loss_cls(pol_logits, batch["polarity"])

        loss = l1 + l2 + l3
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    emo_mae, emp_mae, acc, report = evaluate()
    print(f"\nEpoch {epoch}/{EPOCHS} | train_loss={total_loss/len(train_loader):.4f}")
    print(f"DEV Emotion MAE: {emo_mae:.4f} | DEV Empathy MAE: {emp_mae:.4f} | DEV Polarity Acc: {acc:.4f}")
    print("DEV Polarity report:\n", report)

# Predict test
model.eval()
test_emotion, test_empathy, test_polar = [], [], []
with torch.no_grad():
    for batch in test_loader:
        x = batch["x"]
        emotion, empathy, pol_logits = model(x)
        test_emotion.extend(emotion.numpy().tolist())
        test_empathy.extend(empathy.numpy().tolist())
        test_polar.extend(torch.argmax(pol_logits, dim=1).numpy().tolist())

os.makedirs("outputs", exist_ok=True)
out = pd.DataFrame({
    "id": test["id"].astype(int),
    "Emotion": test_emotion,
    "EmotionalPolarity": test_polar,
    "Empathy": test_empathy
})
out.to_csv("outputs/predictions_ann_sbert.csv", index=False)
print("\nSaved:", "outputs/predictions_ann_sbert.csv")