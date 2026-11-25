"""Webanwendung: Bild hochladen, Münzen erkennen, Ergebnis als HTML-Seite anzeigen."""
import secrets
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

import config
from yolo import Muenzerkenner, format_euro

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
app.jinja_env.filters["euro"] = format_euro
erkenner = Muenzerkenner()


def fehlerseite(text, status=400):
    return render_template("index.html", fehler=text), status


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    datei = request.files.get("bild")
    if not datei or not datei.filename:
        return fehlerseite("Bitte zuerst ein Bild auswählen.")
    endung = Path(datei.filename).suffix.lower()
    if endung not in config.ERLAUBTE_ENDUNGEN:
        return fehlerseite("Nur JPG- und PNG-Dateien werden unterstützt.")

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = secrets.token_hex(8)
    upload = config.UPLOAD_DIR / f"{name}{endung}"
    datei.save(upload)
    try:
        with Image.open(upload) as bild:
            bild.verify()
    except (UnidentifiedImageError, OSError):
        upload.unlink(missing_ok=True)
        return fehlerseite("Die Datei ist kein gültiges Bild.")

    try:
        ergebnis = erkenner.erkenne(upload, config.RESULT_DIR / f"{name}.jpg")
    except FileNotFoundError as fehler:
        return fehlerseite(str(fehler), 503)
    return render_template("index.html", ergebnis=ergebnis, bild=f"{name}.jpg")


@app.route("/ergebnisse/<path:dateiname>")
def ergebnisbild(dateiname):
    return send_from_directory(config.RESULT_DIR, dateiname)


@app.errorhandler(413)
def zu_gross(_):
    return fehlerseite(f"Das Bild ist größer als {config.MAX_UPLOAD_MB} MB.", 413)
