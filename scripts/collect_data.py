"""
Graba secuencias de landmarks para las señas de signs_config.yaml.

Qué cambió respecto de la versión anterior, y por qué (todo salió de medir la
app en un teléfono real):

1. Usa los MISMOS detectores que la app —PoseLandmarker + HandLandmarker de
   MediaPipe Tasks, los .task que viajan dentro del APK— en vez de Holistic.
   Entrenar con un detector y reconocer con otro se paga entero en producción.

2. Graba por TIEMPO, no por cantidad de frames. Antes eran 30 frames a la
   velocidad que diera la laptop, así que nadie sabía cuánto duraba un clip. La
   app toma los últimos 2,5 s y los interpola a 30 pasos; acá se hace igual, así
   que un paso significa lo mismo de los dos lados.

3. Espeja la imagen (cv2.flip) antes de detectar. NO es cosmético: define el
   signo de la coordenada x y qué mano llama "izquierda" MediaPipe. El dataset
   viejo se grabó espejado, así que se mantiene la convención — y la app ahora
   hace lo mismo.

4. Reparte las manos por cercanía al frame anterior en vez de creerle a la
   etiqueta de MediaPipe frame a frame, que con movimiento rápido se equivoca.

5. Guarda metadatos al lado de cada clip: quién firmó, fps reales, duración,
   detector y versión. Sin el nombre del señante no se puede validar agrupando
   por persona, que es la única medida que dice si el modelo le va a servir a
   alguien nuevo.

Uso:
    python scripts/collect_data.py --signante mateo
    python scripts/collect_data.py --sign mama --sequences 30 --signante ana
    python scripts/collect_data.py --list

Salida:
    data/processed/<label>/sequence_<N>.npy        (30, 258)
    data/processed/<label>/sequence_<N>.json       metadatos
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tasks_extractor import (
    AsignadorDeManos,
    armar_keypoints,
    crear_detectores,
)
from src.utils.labels import get_signs, load_config

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELOS = ROOT.parent / "signa-mobile" / "assets" / "mediapipe"

# Lo que dura la ventana que mira la app, y a cuántos pasos la lleva.
VENTANA_S = 2.5
PASOS = 30
VERSION_CAPTURA = 2


def existentes(sign_dir: Path) -> int:
    return len(list(sign_dir.glob("sequence_*.npy")))


def remuestrear(tiempos: list[float], frames: list[np.ndarray]) -> np.ndarray:
    """
    Los frames capturados, repartidos en PASOS instantes equiespaciados.

    Es la misma cuenta que hace la app antes de cada inferencia: así el modelo
    ve la misma escala temporal al entrenar y al reconocer, sin importar a
    cuántos fps corra cada lado.
    """
    t = np.array(tiempos) - tiempos[0]
    objetivo = np.linspace(0, t[-1], PASOS)
    datos = np.array(frames)
    return np.stack(
        [np.interp(objetivo, t, datos[:, c]) for c in range(datos.shape[1])], axis=1
    ).astype(np.float32)


def grabar_sena(sign: dict, args, pose_det, manos_det, reloj) -> int:
    sign_dir = DATA_DIR / sign["folder"]
    sign_dir.mkdir(parents=True, exist_ok=True)

    desde = existentes(sign_dir)
    hasta = desde + args.sequences

    print(f"\n{'=' * 56}")
    print(f"Seña: {sign['label'].upper()}   ·   señante: {args.signante}")
    print(f"Ya grabadas: {desde}   ·   nuevas: {args.sequences}")
    print(f"Cada clip dura {VENTANA_S} s y se guarda con {PASOS} pasos.")
    print(f"{'=' * 56}")
    print("Q para cancelar.\n")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la cámara (índice {args.camera})")
        return 0

    guardadas = 0
    for n in range(desde, hasta):
        # ---- cuenta regresiva ----
        for queda in range(3, 0, -1):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{sign['label']}  ({n + 1}/{hasta})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.putText(frame, f"Empieza en {queda}...", (15, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 255), 2)
            cv2.imshow("Signa - captura", frame)
            if cv2.waitKey(1000) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Cancelado.")
                return guardadas

        # ---- captura por tiempo ----
        asignador = AsignadorDeManos()
        tiempos: list[float] = []
        frames: list[np.ndarray] = []
        inicio = time.perf_counter()

        while True:
            ahora = time.perf_counter()
            transcurrido = ahora - inicio
            if transcurrido >= VENTANA_S:
                break

            ok, frame = cap.read()
            if not ok:
                break
            # Espejado ANTES de detectar: define la x y la mano. Ver cabecera.
            frame = cv2.flip(frame, 1)

            import mediapipe as mp

            imagen = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ms = next(reloj)
            pose_res = pose_det.detect_for_video(imagen, ms)
            manos_res = manos_det.detect_for_video(imagen, ms)
            izq, der = asignador.asignar(manos_res)

            tiempos.append(transcurrido)
            frames.append(armar_keypoints(pose_res, izq, der))

            barra = int(40 * transcurrido / VENTANA_S)
            cv2.putText(frame, f"GRABANDO  {sign['label']}  ({n + 1}/{hasta})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, "[" + "#" * barra + "-" * (40 - barra) + "]", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            manos_vistas = (izq is not None) + (der is not None)
            cv2.putText(frame, f"manos: {manos_vistas}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 0), 2)
            cv2.imshow("Signa - captura", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Cancelado.")
                return guardadas

        # ---- guardar ----
        if len(frames) < PASOS // 2:
            print(f"  ✗ clip {n}: sólo {len(frames)} frames en {VENTANA_S}s, se descarta")
            continue

        secuencia = remuestrear(tiempos, frames)
        np.save(sign_dir / f"sequence_{n}.npy", secuencia)
        (sign_dir / f"sequence_{n}.json").write_text(json.dumps({
            "signante": args.signante,
            "seña": sign["label"],
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "fps_reales": round(len(frames) / VENTANA_S, 1),
            "frames_capturados": len(frames),
            "ventana_s": VENTANA_S,
            "pasos": PASOS,
            "espejado": True,
            "detector": "mediapipe-tasks pose+hands",
            "version_captura": VERSION_CAPTURA,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        guardadas += 1
        print(f"  ✓ clip {n}: {len(frames)} frames capturados "
              f"({len(frames) / VENTANA_S:.0f} fps) → {PASOS} pasos")

    cap.release()
    cv2.destroyAllWindows()
    return guardadas


def reloj_monotono():
    """Timestamps crecientes en ms: MediaPipe en modo VIDEO los exige."""
    t = 0
    while True:
        t += 33
        yield t


def main() -> None:
    p = argparse.ArgumentParser(description="Captura de secuencias de señas.")
    p.add_argument("--sign", help="Grabar sólo esta seña (por label)")
    p.add_argument("--sequences", type=int, help="Clips por seña")
    p.add_argument("--camera", type=int, help="Índice de cámara")
    p.add_argument("--signante", required=False, default="",
                   help="Quién firma. Hace falta para validar agrupando por persona")
    p.add_argument("--models", type=Path, default=MODELOS,
                   help="Carpeta con pose_landmarker.task y hand_landmarker.task")
    p.add_argument("--list", action="store_true", help="Lista las señas del config")
    args = p.parse_args()

    config = load_config()
    signs = get_signs()

    if args.list:
        for s in signs:
            d = DATA_DIR / s["folder"]
            print(f"  {s['label']:<18} {existentes(d) if d.exists() else 0:>3} clips")
        return

    if not args.signante:
        print("ERROR: falta --signante. Sin saber quién firmó cada clip no se")
        print("       puede medir si el modelo le sirve a una persona nueva.")
        sys.exit(1)

    args.sequences = args.sequences or config["collection"]["sequences_per_sign"]
    args.camera = args.camera if args.camera is not None else config["collection"]["camera_index"]

    faltan = [n for n in ("pose_landmarker.task", "hand_landmarker.task")
              if not (args.models / n).exists()]
    if faltan:
        print(f"ERROR: faltan los modelos {faltan} en {args.models}")
        sys.exit(1)

    if args.sign:
        elegida = next((s for s in signs if s["label"] == args.sign), None)
        if not elegida:
            print(f"ERROR: la seña '{args.sign}' no está en el config.")
            sys.exit(1)
        signs = [elegida]

    pose_det, manos_det = crear_detectores(args.models)
    reloj = reloj_monotono()

    total = 0
    for s in signs:
        total += grabar_sena(s, args, pose_det, manos_det, reloj)

    print(f"\n✓ {total} clips nuevos.")
    print("  Ojo: son del extractor nuevo (Tasks). Los clips viejos salieron de")
    print("  Holistic, así que no conviene mezclarlos en un mismo entrenamiento")
    print("  sin comprobar antes que dan lo mismo.")


if __name__ == "__main__":
    main()
