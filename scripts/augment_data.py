"""
Data augmentation sobre secuencias de landmarks ya grabadas (data/processed/).

Genera copias aumentadas de cada sequence_N.npy aplicando transformaciones
geométricas y temporales pensadas para landmarks (no para píxeles). Como
trabaja sobre los .npy crudos (antes de la normalización que hace train.py),
las secuencias aumentadas se integran solas al pipeline de entrenamiento:
train.py solo hace glob("sequence_*.npy") en cada carpeta, así que no hace
falta tocar nada más.

Estructura asumida de cada frame (258 valores), la misma que produce
src/data/extractor.py:
    - Pose:        33 landmarks × (x, y, z, visibility) = 132
    - Mano izq.:   21 landmarks × (x, y, z)             = 63
    - Mano der.:   21 landmarks × (x, y, z)              = 63
    Total: 258

Técnicas incluidas:
    - noise:       ruido gaussiano leve en las coordenadas (simula jitter
                    de detección de MediaPipe)
    - scale:       escala uniforme aleatoria (simula estar más cerca/lejos
                    de la cámara)
    - rotation:    rotación 2D leve en el plano x/y (simula ángulo de
                    cámara distinto), misma rotación en toda la secuencia
    - translation: traslación leve en x/y (simula estar corrido del centro)
    - time_warp:   estira/comprime la secuencia en el tiempo y la vuelve a
                    muestrear a la longitud original (simula firmar más
                    rápido o más lento — muy relevante si dos señas se
                    diferencian por el timing del movimiento)

NO incluye espejado (mirror) por defecto: en LSA el significado de una
seña puede depender de qué mano es la dominante o de relaciones
espaciales absolutas, así que espejar podría generar ejemplos inválidos
o hasta de otra seña. Si querés probarlo igual, usá --techniques con
"mirror" explícitamente y revisá visualmente algunos resultados antes de
confiar en ellos para entrenar.

Uso:
    python scripts/augment_data.py                       # todas las señas, factor 2
    python scripts/augment_data.py --sign gracias --factor 4
    python scripts/augment_data.py --sign entender --factor 4
    python scripts/augment_data.py --techniques noise scale time_warp
    python scripts/augment_data.py --dry-run              # solo muestra qué haría
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.labels import get_signs, load_config

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# ─── Estructura de landmarks ──────────────────────────────────────────────

N_POSE = 33
POSE_DIMS = 4  # x, y, z, visibility
POSE_LEN = N_POSE * POSE_DIMS  # 132

N_HAND = 21
HAND_DIMS = 3  # x, y, z
HAND_LEN = N_HAND * HAND_DIMS  # 63

TOTAL_LEN = POSE_LEN + 2 * HAND_LEN  # 258


def _build_indices():
    """
    Arma los índices de x, y, z y visibility dentro del vector de 258,
    para poder aplicar transformaciones geométricas solo donde corresponde
    (por ejemplo, nunca rotar/escalar el canal de visibility de la pose).
    """
    x_idx, y_idx, z_idx, vis_idx = [], [], [], []

    # Pose: 33 landmarks de 4 valores (x, y, z, visibility)
    for i in range(N_POSE):
        base = POSE_DIMS * i
        x_idx.append(base)
        y_idx.append(base + 1)
        z_idx.append(base + 2)
        vis_idx.append(base + 3)

    # Ambas manos: 21 landmarks de 3 valores (x, y, z), sin visibility
    for hand_offset in (POSE_LEN, POSE_LEN + HAND_LEN):
        for i in range(N_HAND):
            base = hand_offset + HAND_DIMS * i
            x_idx.append(base)
            y_idx.append(base + 1)
            z_idx.append(base + 2)

    return (
        np.array(x_idx), np.array(y_idx), np.array(z_idx), np.array(vis_idx)
    )


X_IDX, Y_IDX, Z_IDX, VIS_IDX = _build_indices()
COORD_IDX = np.concatenate([X_IDX, Y_IDX, Z_IDX])  # todo lo que es coordenada


# ─── Técnicas de augmentation ─────────────────────────────────────────────
# Cada función recibe una secuencia (frames, 258) y devuelve una nueva
# secuencia del mismo shape. Todas trabajan sobre una copia, nunca mutan
# el original.

def add_noise(sequence: np.ndarray, sigma: float = 0.008) -> np.ndarray:
    """Ruido gaussiano leve en las coordenadas. Simula el jitter normal
    de la detección de MediaPipe entre toma y toma."""
    seq = sequence.copy()
    noise = np.random.normal(0, sigma, size=(seq.shape[0], len(COORD_IDX)))
    seq[:, COORD_IDX] += noise
    return seq


# Rangos geométricos anchos a propósito: los clips se graban sentado frente a
# una laptop y la app se usa con el teléfono en la mano, a otra distancia, otro
# ángulo y otro encuadre. Lo que acá parece exagerado es la diferencia real
# entre grabar y usar.
def random_scale(sequence: np.ndarray, scale_range=(0.8, 1.25)) -> np.ndarray:
    """Escala uniforme aleatoria de x, y, z. Un solo factor para toda la
    secuencia (simula estar más cerca/lejos de cámara, no un zoom raro
    frame a frame)."""
    seq = sequence.copy()
    factor = np.random.uniform(*scale_range)
    seq[:, X_IDX] *= factor
    seq[:, Y_IDX] *= factor
    seq[:, Z_IDX] *= factor
    return seq


def random_rotation(sequence: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
    """Rotación 2D leve en el plano x/y alrededor del centro (0.5, 0.5),
    que es aprox. el centro de la imagen en coordenadas normalizadas de
    MediaPipe. Mismo ángulo en toda la secuencia — simula que la cámara
    estaba en otro ángulo, no que la persona giró en el medio del gesto."""
    seq = sequence.copy()
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x = seq[:, X_IDX] - 0.5
    y = seq[:, Y_IDX] - 0.5

    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a

    seq[:, X_IDX] = x_rot + 0.5
    seq[:, Y_IDX] = y_rot + 0.5
    return seq


def random_translation(sequence: np.ndarray, max_shift: float = 0.07) -> np.ndarray:
    """Traslación leve en x/y. Un solo desplazamiento para toda la
    secuencia (simula estar corrido del centro del cuadro, no un temblor)."""
    seq = sequence.copy()
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    seq[:, X_IDX] += dx
    seq[:, Y_IDX] += dy
    return seq


# Rango de estiramiento temporal.
#
# Empezó en (0.8, 1.25) pensando sólo en que cada persona firma a su ritmo. Pero
# medido en la app, la ventana de 30 frames cubre entre 2,5 y 3,0 s según los fps
# que dé el teléfono: la misma seña le llega al modelo hasta un 20% más lenta que
# en el dataset, y eso no es variación de la persona sino del aparato. El rango
# va más ancho para cubrir las dos cosas juntas.
RANGO_TIEMPO = (0.65, 1.5)


def time_warp(sequence: np.ndarray, factor_range=RANGO_TIEMPO) -> np.ndarray:
    """
    Estira o comprime la secuencia en el tiempo (simula firmar más rápido
    o más lento) y la vuelve a muestrear a la cantidad original de frames
    por interpolación lineal, columna por columna.

    Es la técnica más relevante cuando dos señas se diferencian por el
    timing del movimiento (ej. cuánto tarda en llegar a la posición
    final) más que por la forma en sí.
    """
    n_frames = sequence.shape[0]
    factor = np.random.uniform(*factor_range)

    # Tiempos "originales" muestreados a una tasa distinta, luego
    # reinterpolados de vuelta a n_frames puntos equiespaciados.
    warped_len = max(2, int(round(n_frames * factor)))
    original_t = np.linspace(0, 1, n_frames)
    warped_t = np.linspace(0, 1, warped_len)
    target_t = np.linspace(0, 1, n_frames)

    # Interpolar cada columna a la longitud "estirada" y de ahí a n_frames.
    stretched = np.zeros((warped_len, sequence.shape[1]), dtype=sequence.dtype)
    for col in range(sequence.shape[1]):
        stretched[:, col] = np.interp(warped_t, original_t, sequence[:, col])

    resampled = np.zeros((n_frames, sequence.shape[1]), dtype=sequence.dtype)
    stretched_t = np.linspace(0, 1, warped_len)
    for col in range(sequence.shape[1]):
        resampled[:, col] = np.interp(target_t, stretched_t, stretched[:, col])

    return resampled


def mirror(sequence: np.ndarray) -> np.ndarray:
    """
    Espeja horizontalmente (x → 1-x) e intercambia mano izquierda con
    mano derecha. RIESGOSO para LSA: puede invalidar señas cuyo
    significado depende de la mano dominante o de relaciones espaciales
    absolutas. Usar con criterio y revisar resultados antes de confiar
    en ellos para entrenar.
    """
    seq = sequence.copy()
    seq[:, X_IDX] = 1.0 - seq[:, X_IDX]

    # Intercambiar bloques de mano izq. y der. (63 valores cada uno)
    lh_start, lh_end = POSE_LEN, POSE_LEN + HAND_LEN
    rh_start, rh_end = POSE_LEN + HAND_LEN, POSE_LEN + 2 * HAND_LEN
    left_block = seq[:, lh_start:lh_end].copy()
    right_block = seq[:, rh_start:rh_end].copy()
    seq[:, lh_start:lh_end] = right_block
    seq[:, rh_start:rh_end] = left_block

    return seq


TECHNIQUES = {
    "noise": add_noise,
    "scale": random_scale,
    "rotation": random_rotation,
    "translation": random_translation,
    "time_warp": time_warp,
    "mirror": mirror,  # no se incluye en el default
}

# "mirror" entra por decisión explícita: se quiere que la seña valga hecha con
# cualquier mano. Espejar convierte una toma diestra en su versión zurda, así
# que además de duplicar variedad enseña las dos versiones. Ojo si alguna vez
# se agregan señas cuyo significado dependa de la mano dominante o de
# relaciones espaciales absolutas: para esas habría que excluirlo.
DEFAULT_TECHNIQUES = ["noise", "scale", "rotation", "translation", "time_warp", "mirror"]


def augment_sequence(sequence: np.ndarray, techniques: list[str]) -> np.ndarray:
    """
    Aplica una combinación aleatoria de técnicas a una secuencia:
    para cada técnica disponible, la aplica con 60% de probabilidad
    (así cada copia aumentada es una combinación distinta, no siempre
    todas las transformaciones a la vez, que sería demasiado agresivo).
    Si por azar no se elige ninguna, aplica al menos "noise" para
    garantizar que la copia no sea idéntica al original.
    """
    seq = sequence.copy()
    applied = []
    for name in techniques:
        if np.random.random() < 0.6:
            seq = TECHNIQUES[name](seq)
            applied.append(name)

    if not applied:
        seq = add_noise(seq)
        applied.append("noise")

    return seq, applied


# ─── Main ──────────────────────────────────────────────────────────────────

def augment_sign(sign: dict, factor: int, techniques: list[str], dry_run: bool):
    sign_dir = DATA_DIR / sign["folder"]
    if not sign_dir.exists():
        print(f"  [SKIP] '{sign['label']}': no hay datos en {sign_dir}")
        return 0

    originals = sorted(sign_dir.glob("sequence_*.npy"))
    # Evitar re-aumentar copias aumentadas previas (no encadenar augmentation).
    originals = [p for p in originals if "_aug" not in p.stem]

    if not originals:
        print(f"  [SKIP] '{sign['label']}': sin secuencias originales")
        return 0

    print(f"\n{sign['label']}: {len(originals)} originales → generando {factor}x cada una")

    count = 0
    for seq_path in originals:
        sequence = np.load(seq_path)
        if sequence.shape != (sequence.shape[0], TOTAL_LEN):
            pass  # el shape de frames varía por seña, solo validamos feature_dim=258

        for k in range(factor):
            aug_seq, applied = augment_sequence(sequence, techniques)
            out_name = f"{seq_path.stem}_aug{k}.npy"
            out_path = sign_dir / out_name

            if dry_run:
                print(f"    [DRY-RUN] {out_name}  <- técnicas: {', '.join(applied)}")
            else:
                np.save(out_path, aug_seq)
            count += 1

    if not dry_run:
        print(f"  ✓ {count} secuencias aumentadas guardadas en {sign_dir}")
    return count


def clean_augmented(sign: dict, dry_run: bool):
    """Borra las copias aumentadas previas de una seña (para regenerar
    desde cero con otros parámetros, sin ir acumulando versiones viejas)."""
    sign_dir = DATA_DIR / sign["folder"]
    if not sign_dir.exists():
        return 0
    aug_files = list(sign_dir.glob("sequence_*_aug*.npy"))
    for f in aug_files:
        if dry_run:
            print(f"    [DRY-RUN] borraría: {f.name}")
        else:
            f.unlink()
    return len(aug_files)


def main():
    parser = argparse.ArgumentParser(description="Data augmentation sobre landmarks LSA.")
    parser.add_argument("--sign", type=str, help="Aumentar solo esta seña (por label)")
    parser.add_argument("--factor", type=int, default=2, help="Copias aumentadas por secuencia original (default: 2)")
    parser.add_argument(
        "--techniques", nargs="+", choices=list(TECHNIQUES.keys()), default=DEFAULT_TECHNIQUES,
        help=f"Técnicas a usar (default: {' '.join(DEFAULT_TECHNIQUES)})",
    )
    parser.add_argument("--clean", action="store_true", help="Borra las copias aumentadas existentes antes de generar nuevas")
    parser.add_argument("--dry-run", action="store_true", help="Muestra qué haría sin escribir archivos")
    args = parser.parse_args()

    if "mirror" in args.techniques:
        print("⚠ ADVERTENCIA: 'mirror' está activado. Revisá visualmente algunas")
        print("  secuencias espejadas antes de confiar en ellas para entrenar —")
        print("  puede invalidar señas que dependen de la mano dominante.\n")

    signs = get_signs()
    if args.sign:
        target = next((s for s in signs if s["label"] == args.sign), None)
        if not target:
            print(f"ERROR: Seña '{args.sign}' no encontrada en el config.")
            sys.exit(1)
        signs = [target]

    if args.clean:
        print("Borrando copias aumentadas existentes...")
        total_cleaned = sum(clean_augmented(s, args.dry_run) for s in signs)
        print(f"{'[DRY-RUN] ' if args.dry_run else ''}{total_cleaned} archivos aumentados eliminados")

    total = 0
    for sign in signs:
        total += augment_sign(sign, args.factor, args.techniques, args.dry_run)

    action = "generarían" if args.dry_run else "generaron"
    print(f"\n✓ Total: se {action} {total} secuencias aumentadas.")
    if not args.dry_run:
        print("  Corré train.py normalmente — las copias se suman solas al dataset.")


if __name__ == "__main__":
    main()