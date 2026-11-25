import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from PIL import Image

import app as webapp
import config
from yolo import auswerten, bezeichnung, format_euro, wert_in_cent


@pytest.mark.parametrize("name,cent", [
    ("1cent", 1), ("50cent", 50), ("2 Euro", 200), ("1euro", 100),
    ("20_ct", 20), ("5-Cent", 5), ("2€", 200), ("muenze", None),
])
def test_wert_in_cent(name, cent):
    assert wert_in_cent(name) == cent


def test_format_und_bezeichnung():
    assert format_euro(740) == "7,40 €"
    assert format_euro(123456) == "1.234,56 €"
    assert bezeichnung(50) == "50 Cent" and bezeichnung(200) == "2 Euro"


def test_auswerten_beispiel_aus_lastenheft():
    namen = ["2euro"] * 3 + ["50cent"] * 2 + ["10cent"] * 4
    erg = auswerten(namen)
    assert erg["gesamt_anzahl"] == 9
    assert erg["gesamt_cent"] == 740
    assert [p["bezeichnung"] for p in erg["positionen"]] == ["2 Euro", "50 Cent", "10 Cent"]


def test_auswerten_unbekannte_klasse():
    erg = auswerten(["1cent", "xyz"])
    assert erg["gesamt_cent"] == 1 and erg["unbekannt"] == ["xyz"]


class FakeErkenner:
    def erkenne(self, bild, ausgabe):
        Path(ausgabe).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20)).save(ausgabe)
        return auswerten(["2euro", "2euro", "5cent"])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "up")
    monkeypatch.setattr(config, "RESULT_DIR", tmp_path / "res")
    monkeypatch.setattr(webapp, "erkenner", FakeErkenner())
    return webapp.app.test_client()


def bild_bytes(fmt="PNG"):
    puffer = io.BytesIO()
    Image.new("RGB", (30, 30), "white").save(puffer, fmt)
    puffer.seek(0)
    return puffer


def test_upload_zeigt_liste_und_summe(client):
    r = client.post("/", data={"bild": (bild_bytes(), "foto.png")}, content_type="multipart/form-data")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "2 Euro" in html and "5 Cent" in html and "4,05 €" in html
    assert "/ergebnisse/" in html


def test_fehlerfaelle(client):
    assert client.post("/", data={}, content_type="multipart/form-data").status_code == 400
    r = client.post("/", data={"bild": (io.BytesIO(b"x"), "a.gif")}, content_type="multipart/form-data")
    assert r.status_code == 400
    r = client.post("/", data={"bild": (io.BytesIO(b"kein bild"), "a.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_modell_fehlt(client, monkeypatch, tmp_path):
    from yolo import Muenzerkenner
    monkeypatch.setattr(webapp, "erkenner", Muenzerkenner(tmp_path / "gibtsnicht.pt"))
    r = client.post("/", data={"bild": (bild_bytes(), "foto.png")}, content_type="multipart/form-data")
    assert r.status_code == 503 and "train.py" in r.get_data(as_text=True)
