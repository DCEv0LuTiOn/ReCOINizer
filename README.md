# ReCOINizer
KI-Münzerkennung: Münzen auf einem Foto erkennen, zählen und den Gesamtwert berechnen.

## Einrichtung (Windows, PowerShell)
```powershell
uv sync
copy .env.example .env      # danach Roboflow-Zugangsdaten in .env eintragen
```

## Ablauf
```powershell
uv run download_dataset.py  # Datensatz aus Roboflow laden (YOLO-Format)
uv run train.py             # YOLO lokal trainieren -> models/best.pt
uv run evaluate.py          # Accuracy, Precision, Recall, F1, Konfusionsmatrix
uv run main.py              # Webanwendung auf http://127.0.0.1:5000
uv run pytest               # Tests
```

## Dateien
| Datei | Zweck |
|-------|-------|
| `config.py` | Pfade und Einstellungen |
| `download_dataset.py` | Roboflow-Datensatz laden und `data.yaml` reparieren |
| `train.py` | Training des YOLO-Modells |
| `evaluate.py` | Kennzahlen und Konfusionsmatrix |
| `yolo.py` | Erkennung, Zählung, Wertberechnung |
| `app.py`, `templates/index.html` | Upload (JPG/PNG) und HTML-Ausgabe |

Die Klassennamen in Roboflow müssen den Nennwert enthalten, z. B. `1cent`, `50cent`, `2euro`
(auch `2 Euro` oder `50_ct` werden erkannt).
