"""
Exporta el modelo del abecedario a un .tflite que come landmarks crudos.

El modelo entrenado recibe 514 números armados a mano por
`src/data/hand_features.py`: coordenadas canónicas, distancias entre puntos,
ángulos de articulación y la posición de la mano respecto de la cara. Portar
ese cálculo a JavaScript sería ~200 líneas que tienen que dar exactamente lo
mismo que numpy, y ya sabemos lo que cuesta una diferencia mínima entre lo que
ve el modelo al entrenar y al reconocer.

Así que en vez de portarlo, se mete adentro del grafo: el .tflite que va a la
app recibe los landmarks tal como salen de MediaPipe y calcula las features él
mismo. La app no tiene que saber nada de todo esto, y un desajuste pasa a ser
imposible por construcción.

Entradas del modelo exportado:
    lm     (21, 3)  landmarks de imagen, YA espejados si la mano es izquierda
    world  (21, 3)  landmarks métricos, ídem
    cara   (8,)     bloque de posición respecto de la cara (ceros si no hay)

Uso:
    python scripts/export_alphabet_for_app.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hand_features import (FEATURE_DIM, FINGER_CHAINS, INDEX_MCP,
                                    KEY_POINTS, MIDDLE_MCP, N_LANDMARKS,
                                    PINKY_MCP, WRIST, build_features)

ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = ROOT / "models" / "saved_model" / "signa_alphabet.keras"
DEFAULT_OUT = ROOT.parent / "signa-mobile" / "assets" / "tflite"


def _ultimo(patron: str) -> Path:
    archivos = sorted(
        (ROOT / "models" / "exports").glob(patron),
        key=lambda f: int("".join(c for c in f.stem.split("_v")[-1] if c.isdigit()) or 0),
    )
    return archivos[-1]


def capa_de_features(tf):
    """
    Las mismas cuentas que `build_features`, en tensores.

    Única diferencia con numpy: el caso degenerado de la palma vista
    exactamente de canto, donde aquella elige una perpendicular cualquiera y
    acá los ejes quedan en cero. Es un caso de medida nula y en cualquiera de
    las dos versiones las features de ese frame no sirven.

    Se sigue paso por paso la versión de numpy: marco propio de la mano,
    coordenadas canónicas y orientadas, distancias entre los puntos clave y
    ángulos de cada articulación. Cualquier diferencia se ve en la verificación
    del final, que compara contra numpy sobre el dataset real.
    """

    def unit(v, eje=-1):
        n = tf.norm(v, axis=eje, keepdims=True)
        return tf.where(n > 1e-8, v / tf.maximum(n, 1e-12), tf.zeros_like(v))

    def marco(lm):
        v_up = lm[:, MIDDLE_MCP] - lm[:, WRIST]
        escala = tf.norm(v_up, axis=-1, keepdims=True)
        segura = tf.maximum(escala, 1e-12)
        e_y = v_up / segura
        v_across = lm[:, PINKY_MCP] - lm[:, INDEX_MCP]
        proy = tf.reduce_sum(v_across * e_y, axis=-1, keepdims=True) * e_y
        e_x = unit(v_across - proy)
        # El producto cruz a mano: tf.Cross no existe en TFLite.
        e_z = tf.stack([
            e_x[:, 1] * e_y[:, 2] - e_x[:, 2] * e_y[:, 1],
            e_x[:, 2] * e_y[:, 0] - e_x[:, 0] * e_y[:, 2],
            e_x[:, 0] * e_y[:, 1] - e_x[:, 1] * e_y[:, 0],
        ], axis=-1)
        ejes = tf.stack([e_x, e_y, e_z], axis=1)       # (B, 3, 3), ejes como filas
        return ejes, escala

    def canonicas(lm):
        ejes, escala = marco(lm)
        centrado = lm - lm[:, WRIST:WRIST + 1]
        return tf.matmul(centrado, ejes, transpose_b=True) / tf.maximum(
            escala[:, :, None], 1e-12)

    def orientadas(lm):
        _, escala = marco(lm)
        return (lm - lm[:, WRIST:WRIST + 1]) / tf.maximum(escala[:, :, None], 1e-12)

    ii, jj = np.triu_indices(len(KEY_POINTS), k=1)

    def distancias(coords):
        pts = tf.gather(coords, KEY_POINTS, axis=1)
        d = tf.gather(pts, ii, axis=1) - tf.gather(pts, jj, axis=1)
        return tf.norm(d, axis=-1)

    izq = [c[i - 1] for c in FINGER_CHAINS for i in range(1, 4)]
    med = [c[i] for c in FINGER_CHAINS for i in range(1, 4)]
    der = [c[i + 1] for c in FINGER_CHAINS for i in range(1, 4)]

    def angulos(coords):
        a = unit(tf.gather(coords, izq, axis=1) - tf.gather(coords, med, axis=1))
        b = unit(tf.gather(coords, der, axis=1) - tf.gather(coords, med, axis=1))
        return -tf.reduce_sum(a * b, axis=-1)

    def calcular(lm, world):
        ejes, _ = marco(lm)
        canon = canonicas(lm)
        return tf.concat([
            tf.reshape(orientadas(lm), (-1, N_LANDMARKS * 3)),
            tf.reshape(canon, (-1, N_LANDMARKS * 3)),
            tf.reshape(ejes, (-1, 9)),
            distancias(canon),
            angulos(canon),
            tf.reshape(canonicas(world), (-1, N_LANDMARKS * 3)),
        ], axis=1)

    return calcular


def main() -> None:
    p = argparse.ArgumentParser(description="Exporta el abecedario para la app.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--face-repeat", type=int, default=32)
    p.add_argument("--muestras", type=int, default=200,
                   help="Cuántas filas del dataset usar para verificar")
    args = p.parse_args()

    import tensorflow as tf

    clasificador = tf.keras.models.load_model(args.model)
    calcular = capa_de_features(tf)

    lm_in = tf.keras.Input(shape=(N_LANDMARKS, 3), name="landmarks")
    world_in = tf.keras.Input(shape=(N_LANDMARKS, 3), name="world")
    cara_in = tf.keras.Input(shape=(8,), name="cara")

    feats = tf.keras.layers.Lambda(
        lambda t: calcular(t[0], t[1]), output_shape=(FEATURE_DIM,),
        name="features")([lm_in, world_in])
    completo = tf.keras.layers.Concatenate(name="con_cara")(
        [feats] + [cara_in] * args.face_repeat)
    salida = clasificador(completo)
    envoltorio = tf.keras.Model([lm_in, world_in, cara_in], salida)

    # --- verificación contra numpy, que es lo que se usó para entrenar ---
    datos = np.load(ROOT / "data" / "processed_alphabet" / "alphabet_dataset.npz",
                    allow_pickle=True)
    if "lm" in datos:
        lms, worlds = datos["lm"], datos["world"]
    else:
        print("El dataset no guarda los landmarks crudos; se verifica con ruido.")
        rng = np.random.default_rng(0)
        lms = rng.normal(0.5, 0.15, (args.muestras, N_LANDMARKS, 3))
        worlds = rng.normal(0, 0.05, (args.muestras, N_LANDMARKS, 3))

    n = min(args.muestras, len(lms))
    esperado = np.stack([build_features(lms[i], worlds[i]) for i in range(n)])
    obtenido = calcular(tf.constant(lms[:n], tf.float32),
                        tf.constant(worlds[:n], tf.float32)).numpy()
    delta = float(np.abs(esperado - obtenido).max())
    print(f"features en el grafo vs numpy: Δ máx {delta:.2e} sobre {n} muestras")
    if delta > 1e-4:
        print("ERROR: el cálculo dentro del grafo no coincide con el de entrenamiento.")
        sys.exit(1)

    # Vía SavedModel y no desde el modelo de Keras: convertir directo un modelo
    # cargado con BatchNormalization adentro revienta el conversor (falla en
    # MLIR con "missing attribute 'value'"). Exportarlo primero lo evita.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        envoltorio.export(tmp)
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                          tf.lite.OpsSet.SELECT_TF_OPS]
        tflite = conv.convert()

    args.out.mkdir(parents=True, exist_ok=True)
    destino = args.out / "alfabeto.tflite"
    destino.write_bytes(tflite)

    meta = json.loads(_ultimo("signa_alphabet_v*_meta.json").read_text(encoding="utf-8"))
    umbrales = json.loads(
        _ultimo("signa_alphabet_v*_thresholds.json").read_text(encoding="utf-8"))
    etiquetas = [meta["labels"][str(i)] for i in range(len(meta["labels"]))]
    (args.out.parent / "models" / "lsa-alphabet" ).mkdir(parents=True, exist_ok=True)
    (args.out.parent / "models" / "lsa-alphabet" / "manifest.json").write_text(
        json.dumps({
            "labels": etiquetas,
            "thresholds": umbrales["thresholds"],
            "faceRepeat": args.face_repeat,
            "accuracy": meta.get("accuracy"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(tflite) / 1024:.0f} KB → {destino}")
    print(f"  etiquetas: {' '.join(etiquetas)}")


if __name__ == "__main__":
    main()
