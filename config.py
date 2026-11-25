"""Zentrale Einstellungen für ReCOINizer."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Daten und Modelle
DATASET_DIR = BASE_DIR / "data" / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
MODEL_PATH = BASE_DIR / "models" / "best.pt"
RUNS_DIR = BASE_DIR / "runs"

# Webanwendung
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "ergebnisse"
ERLAUBTE_ENDUNGEN = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_MB = 16

# Training und Erkennung
BASIS_MODELL = "yolo26n.pt"   # Vortrainiertes Modell (wird automatisch geladen)
MIN_SICHERHEIT = 0.5          # Mindest-Erkennungssicherheit (conf)
