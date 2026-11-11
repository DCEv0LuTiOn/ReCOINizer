from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# Wert je Klasse in Euro (Klassennamen wie in Roboflow vergeben!)
WERTE = {
    "1cent": 0.01, "2cent": 0.02, "5cent": 0.05,
    "10cent": 0.10, "20cent": 0.20, "50cent": 0.50,
    "1euro": 1.00, "2euro": 2.00,
}

def predict_and_count(image_path):
    
    ergebnis = model.predict("test.jpg", conf=0.5)[0]

    anzahl = {}
    for box in ergebnis.boxes:
        name = model.names[int(box.cls)]
        anzahl[name] = anzahl.get(name, 0) + 1

    gesamt = 0
    print(f"{'Münzart':<10}{'Anzahl':>8}{'Wert':>10}")
    for name, n in anzahl.items():
        wert = n * WERTE[name]
        gesamt += wert
        print(f"{name:<10}{n:>8}{wert:>9.2f} €")

    print(f"{'Gesamt':<10}{sum(anzahl.values()):>8}{gesamt:>9.2f} €")

    # Bild mit eingezeichneten Boxen speichern
    ergebnis.save("ergebnis.jpg")