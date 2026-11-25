"""Bewertet das Modell: Accuracy, Precision, Recall, F1-Score und Konfusionsmatrix."""
import argparse

import numpy as np

import config


def main():
    parser = argparse.ArgumentParser(description="Auswertung des trainierten Modells")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    from ultralytics import YOLO

    modell = YOLO(str(config.MODEL_PATH))
    m = modell.val(
        data=str(config.DATA_YAML),
        split=args.split,
        project=str(config.RUNS_DIR),
        name="auswertung",
        plots=True,  # erzeugt confusion_matrix.png und F1-Kurve
    )
    box, namen = m.box, m.names

    print(f"\n{'Münzart':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for i, klasse in enumerate(box.ap_class_index):
        print(f"{namen[int(klasse)]:<12}{box.p[i]:>10.3f}{box.r[i]:>10.3f}{box.f1[i]:>10.3f}")
    print(f"{'Gesamt':<12}{box.mp:>10.3f}{box.mr:>10.3f}{np.mean(box.f1):>10.3f}")

    # Accuracy: richtig erkannte Münzen / alle Zuordnungen inkl. Fehlern und verpassten Münzen
    matrix = m.confusion_matrix.matrix
    nc = len(namen)
    accuracy = float(np.trace(matrix[:nc, :nc]) / matrix.sum())
    print(f"\nAccuracy:  {accuracy:.3f}")
    print(f"mAP50:     {box.map50:.3f}")
    print(f"Konfusionsmatrix und Diagramme: {m.save_dir}")


if __name__ == "__main__":
    main()
