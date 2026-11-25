"""Trainiert das YOLO-Modell lokal auf dem Münzdatensatz."""
import argparse
import shutil
from pathlib import Path

import config


def main():
    parser = argparse.ArgumentParser(description="YOLO-Training für ReCOINizer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--modell", default=config.BASIS_MODELL)
    args = parser.parse_args()

    if not config.DATA_YAML.exists():
        raise SystemExit("data.yaml fehlt. Bitte zuerst 'uv run download_dataset.py' ausführen.")

    from ultralytics import YOLO

    modell = YOLO(args.modell)  # wird beim ersten Mal automatisch heruntergeladen
    modell.train(
        data=str(config.DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(config.RUNS_DIR),
        name="muenzen",
    )

    beste = Path(modell.trainer.save_dir) / "weights" / "best.pt"
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(beste, config.MODEL_PATH)
    print(f"Bestes Modell kopiert nach: {config.MODEL_PATH}")


if __name__ == "__main__":  # Schutz für Windows (Worker-Prozesse)
    main()
