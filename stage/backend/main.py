from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer
import pickle
import io

app = FastAPI(title="AI Text Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Classe CNN (ordre exact du notebook) ──────────────────
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, pad_idx, embedding_dim,
                 filter_sizes, num_filters, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(embedding_dim, n, f) for f, n in zip(filter_sizes, num_filters)
        ])
        self.fc = nn.Linear(sum(num_filters), num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids):
        x = self.embedding(input_ids).permute(0, 2, 1)
        pooled = [torch.relu(conv(x)).max(dim=2).values for conv in self.conv1d_list]
        return torch.sigmoid(self.fc(self.dropout(torch.cat(pooled, dim=1)))).squeeze(1)

# ── Fix CUDA → CPU ────────────────────────────────────────
class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        if module == '__main__':
            return globals()[name]
        return super().find_class(module, name)

# ── Tokenizer ─────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 512

# ── Chargement CNN ────────────────────────────────────────
cnn_model = None
try:
    with open("saved_models/cnn.pkl", "rb") as f:
        cnn_model = CPUUnpickler(f).load()
    cnn_model.eval()
    print("✅ CNN chargé sur CPU !")
except FileNotFoundError:
    print("⚠️  saved_models/cnn.pkl introuvable")
except Exception as e:
    print(f"❌ Erreur : {e}")

# ── Preprocessing IDENTIQUE à TextDataset ─────────────────
def preprocess(text: str):
    # Nettoyage identique au notebook
    text = " ".join(text.split())

    # encode_plus identique au TextDataset
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True
    )
    ids = inputs['input_ids']

    # Longueur réelle (sans padding) — identique au notebook
    text_length = len([t for t in ids if t != tokenizer.pad_token_id])

    ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    return ids_tensor, text_length

# ── Prédiction ────────────────────────────────────────────
def run_predict(text: str):
    ids, length = preprocess(text)
    with torch.no_grad():
        prob = cnn_model(ids).item()

    print(f"[DEBUG] prob={prob:.6f}  text_length={length}")

    label      = "IA" if prob >= 0.5 else "Humain"
    confidence = round((prob if prob >= 0.5 else 1 - prob) * 100, 1)
    return {
        "label"      : label,
        "confidence" : confidence,
        "probability": round(prob * 100, 1)
    }

# ── Routes ────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running", "model": "CNN", "loaded": cnn_model is not None}

@app.post("/predict")
def predict(req: TextRequest):
    if not req.text.strip():
        return {"error": "Texte vide"}
    if cnn_model is None:
        return {"error": "Modèle non chargé"}
    return run_predict(req.text)

# ── Servir le frontend React ──────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

if os.path.exists("static"):
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        return FileResponse("static/index.html")