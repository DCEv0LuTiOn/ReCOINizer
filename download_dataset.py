"""Lädt den gelabelten Datensatz aus Roboflow im YOLO-Format herunter."""
import os
import sys

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

import config


def hole_einstellung(name):
    wert = os.environ.get(name)
    if not wert:
        sys.exit(f"{name} fehlt. Bitte in der Datei .env eintragen (Vorlage: .env.example).")
    return wert


def data_yaml_reparieren(ordner):
    """Setzt absolute Pfade in der data.yaml, damit Ultralytics den Datensatz sicher findet."""
    datei = ordner / "data.yaml"
    inhalt = yaml.safe_load(datei.read_text(encoding="utf-8"))
    inhalt["path"] = str(ordner.resolve())
    inhalt["train"] = "train/images"
    inhalt["val"] = "valid/images"
    if (ordner / "test" / "images").exists():
        inhalt["test"] = "test/images"
    datei.write_text(yaml.safe_dump(inhalt, allow_unicode=True), encoding="utf-8")
    return inhalt


def main():
    load_dotenv()
    rf = Roboflow(api_key=hole_einstellung("ROBOFLOW_API_KEY"))
    projekt = rf.workspace(hole_einstellung("ROBOFLOW_WORKSPACE")).project(
        hole_einstellung("ROBOFLOW_PROJECT")
    )
    version = int(hole_einstellung("ROBOFLOW_VERSION"))
    datensatz = projekt.version(version).download("yolov8", location=str(config.DATASET_DIR))

    inhalt = data_yaml_reparieren(config.DATASET_DIR)
    print(f"Datensatz (Version {version}) liegt in: {datensatz.location}")
    print(f"Klassen ({len(inhalt['names'])}): {inhalt['names']}")


if __name__ == "__main__":
    main()
