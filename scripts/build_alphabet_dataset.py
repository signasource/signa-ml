"""
Convierte las fotos del abecedario LSA en un dataset de features de mano.

Entrada:
    data/raw/abc/**/LSA_<LETRA>_<FUENTE>.png|jpg|jpeg

    <FUENTE> es el id del video del que se sacó el screenshot (ver el Tracker
    en Drive). NO es un índice de muestra: cada letra tiene ~1 foto por fuente,
    y cada fuente es un señante distinto. Por eso lo guardamos: es el grupo
    contra el que después se hace el split en train_alphabet.py.

    Las subcarpetas dentro de abc/ no importan: se busca recursivamente y la
    letra sale del nombre del archivo.

Salida:
    data/processed_alphabet/alphabet_dataset.npz
    reports/alphabet_extraction.csv   (una fila por imagen, detectada o no)

Cada imagen original genera 1 muestra + N variantes aumentadas. La aumentación
se hace ANTES de MediaPipe, a propósito: así el dataset incluye el error real
del detector ante rotaciones e iluminación distintas, que es la principal
fuente de fallos en vivo.

Uso:
    python scripts/build_alphabet_dataset.py
    python scripts/build_alphabet_dataset.py --no-augment
    python scripts/build_alphabet_dataset.py --variants 10
    python scripts/build_alphabet_dataset.py --raw-dir data/raw/abc --debug-dir reports/debug
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hand_features import (build_features, build_face_features, mirror,
                                    FEATURE_DIM, FACE_BLOCK_DIM)

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "alphabet_config.yaml"
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "abc"
OUT_DIR = ROOT / "data" / "processed_alphabet"
REPORTS_DIR = ROOT / "reports"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# LSA_A_001 · LSA_S_0015 · LSA_V_014_2 (segunda toma de la misma fuente)
NAME_RE = re.compile(r"^LSA[_-](.+?)[_-]0*(\d+)(?:[_-](\d+))?$", re.IGNORECASE)


# ─── Parseo de nombres ───────────────────────────────────────────────────────

def parse_name(stem: str) -> tuple[str, int] | None:
    """'LSA_A_001' → ('A', 1). Devuelve None si el nombre no matchea."""
    m = NAME_RE.match(unicodedata.normalize("NFC", stem).strip())
    if not m:
        return None
    letter = unicodedata.normalize("NFC", m.group(1)).upper()
    return letter, int(m.group(2))


# ─── Aumentación a nivel imagen ──────────────────────────────────────────────

def augment_image(img: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]

    angle = rng.uniform(-cfg["rotation_deg"], cfg["rotation_deg"])
    scale = rng.uniform(*cfg["scale"])
    tx = rng.uniform(-cfg["translate_frac"], cfg["translate_frac"]) * w
    ty = rng.uniform(-cfg["translate_frac"], cfg["translate_frac"]) * h

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    # BORDER_REPLICATE en vez de negro: un borde negro duro le inventa
    # contornos al detector y le hace perder la mano cerca del margen.
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    alpha = rng.uniform(*cfg["contrast"])
    beta = (rng.uniform(*cfg["brightness"]) - 1.0) * 60.0
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < cfg["blur_prob"]:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)

    if rng.random() < cfg["jpeg_prob"]:
        quality = int(rng.integers(35, 80))
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return out


# ─── Extracción ──────────────────────────────────────────────────────────────

def best_hand(results):
    """
    De todas las manos detectadas devuelve la de mayor score, como
    (landmarks (21,3), world_landmarks (21,3) | None, label, score).
    """
    if not results.multi_hand_landmarks:
        return None

    handedness = results.multi_handedness or []
    scores = [h.classification[0].score for h in handedness] or [0.0] * len(results.multi_hand_landmarks)
    idx = int(np.argmax(scores))

    lm = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[idx].landmark])

    world = None
    if getattr(results, "multi_hand_world_landmarks", None):
        world = np.array([[p.x, p.y, p.z]
                          for p in results.multi_hand_world_landmarks[idx].landmark])

    label = handedness[idx].classification[0].label if idx < len(handedness) else "Right"
    return lm, world, label, float(scores[idx])


def face_reference(rgb: np.ndarray, face):
    """
    (centro de los ojos, centro de la boca) en coordenadas relativas, o (None, None).

    Se usa BlazeFace (`FaceDetection`), no FaceMesh: pesa ~200 KB, da los 6
    puntos que necesitamos y corre en pocos milisegundos al lado de Hands.
    """
    if face is None:
        return None, None
    import mediapipe as mp          # se importa tarde, igual que en main()
    det = face.process(rgb)
    if not det.detections:
        return None, None
    kp = det.detections[0].location_data.relative_keypoints
    KP = mp.solutions.face_detection.FaceKeyPoint
    ojo = np.array([(kp[KP.RIGHT_EYE].x + kp[KP.LEFT_EYE].x) / 2,
                    (kp[KP.RIGHT_EYE].y + kp[KP.LEFT_EYE].y) / 2])
    boca = np.array([kp[KP.MOUTH_CENTER].x, kp[KP.MOUTH_CENTER].y])
    return ojo, boca


def extract(img: np.ndarray, hands, mirror_left: bool, face=None):
    """
    Corre los detectores sobre una imagen BGR y devuelve
    (features, cara, label, score) o None.

    Usa MediaPipe Tasks —los mismos .task que viajan en la app— y no las
    soluciones legacy con las que se armó la primera versión del dataset. Son
    modelos distintos: entrenar con unos y reconocer con otros es el desajuste
    que ya nos costó un día entero de depuración.
    """
    import mediapipe as mp
    from src.data.tasks_extractor import mano_principal, referencia_cara

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    found = mano_principal(hands.detect(imagen))
    if found is None:
        return None

    lm, world, label, score = found
    espejada = mirror_left and label.lower().startswith("l")

    # El bloque de cara se calcula con los landmarks SIN espejar y se le pasa el
    # flag: espejar la mano sin espejar su posición daría signos cruzados.
    ojo, boca = referencia_cara(face.detect(imagen)) if face is not None else (None, None)
    cara = build_face_features(lm, ojo, boca, mirrored=espejada)

    if espejada:
        lm = mirror(lm)
        if world is not None:
            world = mirror(world)

    return build_features(lm, world), cara, label, score


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Construye el dataset de features del abecedario.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "alphabet_dataset.npz")
    parser.add_argument("--variants", type=int, default=None,
                        help="Variantes aumentadas por imagen (default: el config)")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="Si se pasa, guarda ahí las imágenes donde NO se detectó mano")
    parser.add_argument("--models", type=Path,
                        default=Path(__file__).parent.parent.parent / "signa-mobile"
                        / "assets" / "mediapipe",
                        help="Carpeta con pose_landmarker.task y hand_landmarker.task")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    ex_cfg = config["extraction"]
    aug_cfg = ex_cfg["augment"]

    variants = 0 if args.no_augment else (
        args.variants if args.variants is not None
        else (aug_cfg["variants_per_image"] if aug_cfg["enabled"] else 0)
    )

    letters = [unicodedata.normalize("NFC", l).upper() for l in config["letters"]]
    letter_to_id = {l: i for i, l in enumerate(letters)}

    if not args.raw_dir.exists():
        print(f"ERROR: no existe {args.raw_dir}")
        print("Descargá la carpeta 'Letras' de Drive y descomprimila ahí dentro.")
        sys.exit(1)

    images = sorted(p for p in args.raw_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"ERROR: no hay imágenes en {args.raw_dir}")
        sys.exit(1)

    print(f"Imágenes encontradas: {len(images)}")
    print(f"Variantes aumentadas por imagen: {variants}")

    import mediapipe as mp

    rng = np.random.default_rng(args.seed)
    X, F, y, sources, is_aug, mirrored = [], [], [], [], [], []
    mirror_images = bool(aug_cfg.get("mirror_images", False))
    mirror_variants = 0 if args.no_augment else int(aug_cfg.get("mirror_variants", variants))
    rows = []
    unparsed, unknown_letter, undetected = [], [], []

    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    from src.data.tasks_extractor import detectores_estaticos

    # La referencia de cara sale de la pose, no de BlazeFace: es un detector
    # menos y es el que la app ya corre.
    face, hands = detectores_estaticos(args.models)
    with face, hands:

        for n, path in enumerate(images, 1):
            parsed = parse_name(path.stem)
            if parsed is None:
                unparsed.append(path.name)
                continue

            letter, source = parsed
            if letter not in letter_to_id:
                unknown_letter.append(f"{path.name} → '{letter}'")
                continue

            img = cv2.imread(str(path))
            if img is None:
                rows.append([path.name, letter, source, 0, "", "", "no_se_pudo_leer"])
                continue

            # Imágenes gigantes ralentizan MediaPipe sin aportar precisión.
            if max(img.shape[:2]) > 1280:
                f = 1280 / max(img.shape[:2])
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

            # La cámara de la demo y de la app llega ESPEJADA, y las fotos son
            # capturas de video sin espejar. MediaPipe no es simétrico: la foto
            # espejada no da los landmarks espejados, y en letras que se deciden
            # por milímetros (T/I) eso alcanzaba para cambiar la letra. Así que
            # cada foto entra también espejada. Cuenta como original
            # (is_augmented = 0): es una foto real, vista como la ve la cámara.
            for flip in ((False, True) if mirror_images else (False,)):
                base = cv2.flip(img, 1) if flip else img
                n_var = mirror_variants if flip else variants
                for v in range(n_var + 1):
                    frame = base if v == 0 else augment_image(base, aug_cfg, rng)
                    got = extract(frame, hands, ex_cfg["mirror_left_to_right"], face)

                    if got is None:
                        if v == 0 and not flip:
                            undetected.append(path.name)
                            rows.append([path.name, letter, source, 0, "", "", "sin_mano"])
                            if args.debug_dir:
                                cv2.imwrite(str(args.debug_dir / path.name), img)
                        continue

                    features, cara, label, score = got
                    X.append(features)
                    F.append(cara)
                    y.append(letter_to_id[letter])
                    sources.append(source)
                    is_aug.append(1 if v else 0)
                    mirrored.append(1 if flip else 0)

                    if v == 0 and not flip:
                        rows.append([path.name, letter, source, 1, label, f"{score:.3f}", "ok"])

            if n % 25 == 0 or n == len(images):
                print(f"  {n}/{len(images)} imágenes · {len(X)} muestras")

    if not X:
        print("ERROR: no se extrajo ninguna mano. Revisá reports/alphabet_extraction.csv")
        sys.exit(1)

    X = np.stack(X).astype(np.float32)
    F = np.stack(F).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    sources = np.array(sources, dtype=np.int64)
    is_aug = np.array(is_aug, dtype=np.int8)
    mirrored = np.array(mirrored, dtype=np.int8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X, face=F, y=y, sources=sources, is_augmented=is_aug, mirrored=mirrored,
        letters=np.array(letters, dtype=object),
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "alphabet_extraction.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "letra", "fuente", "detectada", "mano", "score", "estado"])
        w.writerows(rows)

    # ─── Resumen ─────────────────────────────────────────────────────────────
    originals = int((is_aug == 0).sum())
    sin_cara = int((F[:, -1] == 0).sum())
    print(f"\nDataset: {X.shape[0]} muestras × {FEATURE_DIM} features de mano"
          f" + {FACE_BLOCK_DIM} de posición respecto de la cara")
    if sin_cara:
        print(f"  sin cara detectada en {sin_cara} ({sin_cara / len(F):.0%}): "
              f"ese bloque va en cero y el modelo lo ignora")
    print(f"  originales: {originals} (espejadas: {int(((is_aug == 0) & (mirrored == 1)).sum())})"
          f"  |  aumentadas: {X.shape[0] - originals}")
    print(f"  fuentes distintas: {len(set(sources.tolist()))}")
    print(f"Guardado en: {args.out}")
    print(f"Reporte:     {report_path}")

    print("\nMuestras ORIGINALES por letra (las aumentadas son copias, no cuentan):")
    missing = []
    for i, letter in enumerate(letters):
        n_orig = int(((y == i) & (is_aug == 0) & (mirrored == 0)).sum())
        n_src = len(set(sources[(y == i)].tolist()))
        flag = "  ← POCAS" if n_orig < 8 else ""
        if n_orig == 0:
            missing.append(letter)
        print(f"  {letter:>2}: {n_orig:>3} fotos · {n_src:>2} fuentes{flag}")

    if missing:
        print(f"\n⚠ Sin datos: {', '.join(missing)} — se van a excluir al entrenar.")
    if undetected:
        print(f"\n⚠ {len(undetected)} fotos sin mano detectada:")
        for name in undetected[:20]:
            print(f"    {name}")
        if len(undetected) > 20:
            print(f"    … y {len(undetected) - 20} más (todas en el CSV)")
        print("  Mirálas con --debug-dir reports/debug para ver si son recortes malos.")
    if unparsed:
        print(f"\n⚠ {len(unparsed)} archivos con nombre fuera de convención (ignorados):")
        for name in unparsed[:15]:
            print(f"    {name}")
    if unknown_letter:
        print(f"\n⚠ {len(unknown_letter)} archivos con letra que no está en el config:")
        for name in unknown_letter[:15]:
            print(f"    {name}")


if __name__ == "__main__":
    main()
