"""Münzerkennung mit YOLO: erkennen, zählen, Wert berechnen."""
import re
from collections import Counter
from pathlib import Path

from PIL import Image

import config

_KLASSE = re.compile(r"(\d+)(cent|ct|euro|eur|€)")


def wert_in_cent(klassenname):
    """Wandelt einen Klassennamen (z. B. '2cent', '2 Euro', '50_ct') in Cent um.

    Gibt None zurück, wenn der Name nicht erkannt wird.
    """
    name = re.sub(r"[\s_\-.]", "", str(klassenname).lower())
    treffer = _KLASSE.fullmatch(name)
    if not treffer:
        return None
    zahl, einheit = int(treffer.group(1)), treffer.group(2)
    return zahl if einheit in ("cent", "ct") else zahl * 100


def bezeichnung(cent):
    """Anzeigename einer Münze, z. B. '50 Cent' oder '2 Euro'."""
    if cent >= 100 and cent % 100 == 0:
        return f"{cent // 100} Euro"
    return f"{cent} Cent"


def format_euro(cent):
    """Formatiert Cent als deutschen Euro-Betrag, z. B. 740 -> '7,40 €'."""
    euro, rest = divmod(int(cent), 100)
    return f"{euro:,}".replace(",", ".") + f",{rest:02d} €"


def auswerten(klassennamen):
    """Zählt die erkannten Klassen und berechnet Einzel- und Gesamtwerte."""
    positionen, unbekannt = [], []
    for name, anzahl in Counter(klassennamen).items():
        cent = wert_in_cent(name)
        if cent is None:
            unbekannt.append(name)
            continue
        positionen.append({
            "klasse": name,
            "bezeichnung": bezeichnung(cent),
            "anzahl": anzahl,
            "einzelwert_cent": cent,
            "summe_cent": anzahl * cent,
        })
    positionen.sort(key=lambda p: p["einzelwert_cent"], reverse=True)
    return {
        "positionen": positionen,
        "gesamt_anzahl": sum(p["anzahl"] for p in positionen),
        "gesamt_cent": sum(p["summe_cent"] for p in positionen),
        "unbekannt": sorted(unbekannt),
    }


class Muenzerkenner:
    """Lädt das trainierte YOLO-Modell beim ersten Einsatz."""

    def __init__(self, modell_pfad=config.MODEL_PATH, conf=config.MIN_SICHERHEIT):
        self.modell_pfad = Path(modell_pfad)
        self.conf = conf
        self._modell = None

    def _lade_modell(self):
        if self._modell is None:
            if not self.modell_pfad.exists():
                raise FileNotFoundError(
                    f"Kein trainiertes Modell gefunden: {self.modell_pfad}. "
                    "Bitte zuerst 'uv run train.py' ausführen."
                )
            from ultralytics import YOLO  # erst hier, damit der Import schnell bleibt
            self._modell = YOLO(str(self.modell_pfad))
        return self._modell

    def erkenne(self, bild_pfad, ausgabe_pfad=None):
        """Erkennt Münzen auf einem Bild und gibt das ausgewertete Ergebnis zurück.

        Mit ausgabe_pfad wird zusätzlich das Bild mit Erkennungsboxen gespeichert.
        """
        modell = self._lade_modell()
        ergebnisse = list(modell.predict(str(bild_pfad), conf=self.conf, verbose=False))
        erg = ergebnisse[0]
        namen = [modell.names[int(box.cls)] for box in erg.boxes]
        if ausgabe_pfad is not None:
            Path(ausgabe_pfad).parent.mkdir(parents=True, exist_ok=True)
            bgr = erg.plot()  # numpy-Array in BGR
            Image.fromarray(bgr[..., ::-1]).save(ausgabe_pfad)
        return auswerten(namen)


def main():
    import sys
    if len(sys.argv) < 2:
        sys.exit("Aufruf: uv run yolo.py <bild.jpg>")
    erg = Muenzerkenner().erkenne(sys.argv[1], config.RESULT_DIR / "ergebnis.jpg")
    print(f"{'Münzart':<10}{'Anzahl':>8}{'Wert':>12}")
    for p in erg["positionen"]:
        print(f"{p['bezeichnung']:<10}{p['anzahl']:>8}{format_euro(p['summe_cent']):>12}")
    print(f"{'Gesamt':<10}{erg['gesamt_anzahl']:>8}{format_euro(erg['gesamt_cent']):>12}")


if __name__ == "__main__":
    main()
