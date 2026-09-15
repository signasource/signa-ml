"""
Graba fotos del abecedario desde la webcam, con la convención de nombres del
pipeline, para sumarlas al dataset de Drive.

Por qué importa: las fotos de Drive son screenshots de 15 videos de terceros
(otra cámara, otra luz, otra distancia). Si el modelo sólo ve eso, en tu webcam
va a andar peor. Grabando 20-30 tomas por letra con la cámara y la persona que
van a salir en el video, el modelo aprende también ese dominio — es lo que más
sube la precisión de la demo por unidad de esfuerzo.

Las fotos se guardan como data/raw/abc/webcam_<perfil>/LSA_<LETRA>_<FUENTE>.png
con un id de fuente propio (default 100+), para que no choque con los ids del
Tracker y el split por fuente los siga tratando como un grupo aparte.

Uso:
    python scripts/collect_alphabet.py --profile mateo
    python scripts/collect_alphabet.py --profile mateo --letters A B C --shots 30
    python scripts/collect_alphabet.py --profile cande --source-id 200

Controles:
    ESPACIO → empezar a capturar la letra actual
    N / P   → siguiente / anterior letra
    R       → borrar lo capturado de esta letra
    Q       → salir
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "alphabet_config.yaml"


def maybe_fix_linux_qt_backend():
    if platform.system() == "Linux" and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    all_letters = [unicodedata.normalize("NFC", l).upper() for l in config["letters"]]

    p = argparse.ArgumentParser(description="Captura fotos del abecedario desde la webcam.")
    p.add_argument("--profile", required=True, help="Quién graba (va en el nombre de la carpeta)")
    p.add_argument("--letters", nargs="*", default=None, help="Subconjunto de letras")
    p.add_argument("--shots", type=int, default=25, help="Fotos por letra")
    p.add_argument("--interval", type=float, default=0.35, help="Segundos entre fotos")
    p.add_argument("--source-id", type=int, default=100,
                   help="Id de fuente base; cada letra usa source-id + índice de toma")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    letters = [unicodedata.normalize("NFC", l).upper() for l in (args.letters or all_letters)]
    unknown = [l for l in letters if l not in all_letters]
    if unknown:
        print(f"ERROR: letras fuera del config: {', '.join(unknown)}")
        sys.exit(1)

    out_dir = ROOT / "data" / "raw" / "abc" / f"webcam_{args.profile}"
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_fix_linux_qt_backend()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la cámara {args.camera}")
        sys.exit(1)

    print(f"Guardando en: {out_dir}")
    print("ESPACIO capturar · N/P cambiar letra · R rehacer · Q salir\n")

    i = 0
    capturing = False
    taken = 0
    last_shot = 0.0

    def existing(letter):
        return sorted(out_dir.glob(f"LSA_{letter}_*.png"))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            letter = letters[i]
            have = len(existing(letter))

            if capturing and time.time() - last_shot >= args.interval:
                # Cada foto es su propia "fuente": son tomas independientes de
                # la misma persona, y así el split las puede repartir.
                source = args.source_id + have
                cv2.imwrite(str(out_dir / f"LSA_{letter}_{source:03d}.png"), frame)
                last_shot = time.time()
                taken += 1
                have += 1
                if taken >= args.shots:
                    capturing = False
                    taken = 0
                    print(f"  ✓ {letter}: {have} fotos")

            banner = frame.copy()
            cv2.rectangle(banner, (0, 0), (w, 64), (28, 24, 20), -1)
            cv2.addWeighted(banner, 0.8, frame, 0.2, 0, dst=frame)

            cv2.putText(frame, letter, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.6, (76, 195, 247), 3, cv2.LINE_AA)
            cv2.putText(frame, f"{have} fotos  ({i + 1}/{len(letters)})", (100, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

            if capturing:
                cv2.circle(frame, (w - 40, 32), 12, (60, 60, 240), -1, cv2.LINE_AA)
                cv2.putText(frame, f"REC {taken}/{args.shots}", (w - 210, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 240), 2, cv2.LINE_AA)
                cv2.putText(frame, "movete un poco: angulo, distancia, altura",
                            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "ESPACIO para capturar", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow("Signa - Captura de abecedario", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                capturing = not capturing
                taken = 0
                last_shot = 0.0
            elif key in (ord("n"), 83):
                i = (i + 1) % len(letters)
                capturing = False
                taken = 0
            elif key in (ord("p"), 81):
                i = (i - 1) % len(letters)
                capturing = False
                taken = 0
            elif key == ord("r"):
                for f in existing(letter):
                    f.unlink()
                print(f"  ✗ {letter}: borradas")
                capturing = False
                taken = 0
    finally:
        cap.release()
        cv2.destroyAllWindows()

    total = len(list(out_dir.glob("LSA_*.png")))
    print(f"\nTotal en {out_dir.name}: {total} fotos")
    print("Ahora corré: python scripts/build_alphabet_dataset.py && python scripts/train_alphabet.py --cv")


if __name__ == "__main__":
    main()
