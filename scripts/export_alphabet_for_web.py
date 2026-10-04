"""
Exporta el abecedario para la web (la demo de cámara de signa-web), sin TFLite.

La landing no puede usar el runtime de TFLite para web: su cargador de
Emscripten usa `eval`, y el CSP del sitio no lo permite (comparte origen con el
panel, donde vive el token). Así que la red se exporta como pesos crudos y se
corre en TypeScript, igual que se hizo con la LSTM de señas (export_for_app.py).

Cada red del ensemble es Normalization → 3×(Dense sin bias, BatchNorm, ReLU) →
Dense softmax. Al predecir, la normalización y los BatchNorm son afines, así
que se pliegan adentro de las Dense: queda multiplicar matrices + ReLU, y la
salida es la misma. Además el bloque de cara entra repetido `face_repeat`
veces con los mismos valores: sus filas de la primera capa se suman y la
entrada pasa de 514 a 258 + 8.

Salida (en signa-web):
    public/reconocedor/alfabeto.bin   float32 little-endian, capa por capa (W fila por fila, después b)
    public/reconocedor/alfabeto.json  etiquetas, umbrales y forma de cada capa
    src/lib/alphabet-classifier.vectors.json
        casos reales (landmarks de fotos del dataset pasadas por los detectores de
        la app) con las features y probabilidades que da Python: los tests de
        signa-web comparan la versión TypeScript contra esto.

Uso:
    python scripts/export_alphabet_for_web.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.hand_features import (FACE_BLOCK_DIM, FEATURE_DIM,  # noqa: E402
                                    build_face_features, build_features, mirror)

DEFAULT_MODEL = ROOT / "models" / "saved_model" / "signa_alphabet.keras"
WEB = ROOT.parent / "signa-web"
MEDIAPIPE = ROOT.parent / "signa-mobile" / "assets" / "mediapipe"


def _ultimo(patron: str) -> Path:
    archivos = sorted((ROOT / "models" / "exports").glob(patron),
                      key=lambda f: int("".join(c for c in f.stem.split("_v")[-1] if c.isdigit()) or 0))
    return archivos[-1]


def plegar(red, face_repeat: int) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Las capas de una red como [(W, b, relu)], con normalización y BatchNorm adentro."""
    import tensorflow as tf

    capas = [l for l in red.layers]
    norm = next(l for l in capas if isinstance(l, tf.keras.layers.Normalization))
    mean, var = norm.mean.numpy().ravel(), norm.variance.numpy().ravel()
    sigma = np.maximum(np.sqrt(var), 1e-7)       # lo mismo que hace la capa de Keras
    # 48 de las 514 entradas son constantes en todo el dataset (la muñeca, que
    # es el origen de las coordenadas canónicas, y compañía): varianza 0. Keras
    # las divide por 1e-7, y como al entrenar siempre valen su media, aportan 0.
    # Plegado, eso es multiplicar por 10 millones y restar lo mismo: la
    # cancelación se come la precisión, y en vivo un redondeo de 1e-7 en esa
    # entrada se volvería una unidad entera de señal. Se les da peso 0, que es
    # exactamente lo que aportan cuando valen su constante.
    constante = var < 1e-12

    salida: list[tuple[np.ndarray, np.ndarray, bool]] = []
    W = b = None
    primera = True
    for l in capas:
        if isinstance(l, tf.keras.layers.Dense):
            if W is not None:
                salida.append((W, b, True))
            pesos = l.get_weights()
            W = pesos[0].astype(np.float64)
            b = pesos[1].astype(np.float64) if l.use_bias else np.zeros(W.shape[1])
            if primera:
                # (x - mean) / sigma  →  W / sigma  y  b - (mean / sigma) @ W
                W[constante] = 0.0
                b = b - (mean / sigma) @ W
                W = W / sigma[:, None]
                primera = False
            if l.get_config()["activation"] == "softmax":
                salida.append((W, b, False))
                W = None
        elif isinstance(l, tf.keras.layers.BatchNormalization):
            gamma, beta, mu, v = (w.astype(np.float64) for w in l.get_weights())
            k = gamma / np.sqrt(v + l.epsilon)
            W = W * k[None, :]
            b = (b - mu) * k + beta

    # Las `face_repeat` copias del bloque de cara reciben siempre los mismos 8
    # valores: sumar sus filas da exactamente lo mismo con 8 entradas en vez de 256.
    W0, b0, r0 = salida[0]
    cara = W0[FEATURE_DIM:].reshape(face_repeat, FACE_BLOCK_DIM, -1).sum(axis=0)
    salida[0] = (np.vstack([W0[:FEATURE_DIM], cara]), b0, r0)
    return salida


def adelante(redes, x: np.ndarray) -> np.ndarray:
    """El forward que va a hacer TypeScript, en numpy: para verificar el plegado."""
    probs = []
    for capas in redes:
        h = x
        for W, b, relu in capas:
            h = h @ W + b
            h = np.maximum(h, 0) if relu else h
        e = np.exp(h - h.max(axis=-1, keepdims=True))
        probs.append(e / e.sum(axis=-1, keepdims=True))
    return np.mean(probs, axis=0)


def casos_reales(n_por_letra: int, letras: list[str]):
    """Landmarks de fotos del dataset por los detectores de la app, en las dos orientaciones."""
    import cv2
    import mediapipe as mp
    from src.data.tasks_extractor import detectores_estaticos, mano_principal, referencia_cara

    pose, hands = detectores_estaticos(MEDIAPIPE)
    casos = []
    for letra in letras:
        fotos = sorted(glob.glob(str(ROOT / "data" / "raw" / "abc" / "**" / f"LSA_{letra}_*.*"),
                            recursive=True))[:n_por_letra]
        for i, f in enumerate(fotos):
            img = cv2.imread(f)
            if img is None:
                continue
            if i % 2:
                img = cv2.flip(img, 1)   # la mitad espejada: así hay manos izquierdas
            imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            found = mano_principal(hands.detect(imagen))
            pres = pose.detect(imagen)
            if found is None or not pres.pose_landmarks:
                continue
            lm, world, label, _ = found
            casos.append({
                "letra": letra,
                "lm": lm.round(6).tolist(),
                "world": (world if world is not None else np.zeros((21, 3))).round(6).tolist(),
                "pose": [[round(p.x, 6), round(p.y, 6)] for p in pres.pose_landmarks[0]],
                "mirrored": label.lower().startswith("l"),
                "_ref": referencia_cara(pres),
            })
    return casos


def main() -> None:
    p = argparse.ArgumentParser(description="Exporta el abecedario para la demo web.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--web", type=Path, default=WEB)
    p.add_argument("--face-repeat", type=int, default=32)
    p.add_argument("--casos-por-letra", type=int, default=2)
    args = p.parse_args()

    import tensorflow as tf
    from src.utils.losses import CUSTOM_OBJECTS

    modelo = tf.keras.models.load_model(args.model, compile=False, custom_objects=CUSTOM_OBJECTS)
    redes = [plegar(l, args.face_repeat) for l in modelo.layers
             if isinstance(l, tf.keras.Model)] or [plegar(modelo, args.face_repeat)]

    meta = json.loads(_ultimo("signa_alphabet_v*_meta.json").read_text(encoding="utf-8"))
    umbrales = json.loads(_ultimo("signa_alphabet_v*_thresholds.json").read_text(encoding="utf-8"))
    etiquetas = [meta["labels"][str(i)] for i in range(len(meta["labels"]))]

    # ── Verificación del plegado contra Keras, sobre el dataset real ──
    d = np.load(ROOT / "data" / "processed_alphabet" / "alphabet_dataset.npz", allow_pickle=True)
    X = np.hstack([d["X"]] + [d["face"]] * args.face_repeat).astype(np.float32)[:400]
    keras_p = modelo.predict(X, verbose=0)
    corto = np.hstack([X[:, :FEATURE_DIM], X[:, FEATURE_DIM:FEATURE_DIM + FACE_BLOCK_DIM]])
    delta = float(np.abs(adelante(redes, corto.astype(np.float64)) - keras_p).max())
    print(f"plegado vs Keras: Δ máx {delta:.1e} sobre {len(X)} muestras")
    if delta > 1e-4:
        sys.exit("ERROR: el modelo plegado no da lo mismo que Keras.")

    # ── Pesos ──
    out = args.web / "public" / "reconocedor"
    out.mkdir(parents=True, exist_ok=True)
    capas_json, blobs = [], []
    for capas in redes:
        capas_json.append([{"in": int(W.shape[0]), "out": int(W.shape[1]), "relu": relu}
                           for W, b, relu in capas])
        for W, b, _ in capas:
            blobs += [W.astype("<f4").tobytes(), b.astype("<f4").tobytes()]
    (out / "alfabeto.bin").write_bytes(b"".join(blobs))
    (out / "alfabeto.json").write_text(json.dumps({
        "model": f"signa_alphabet_v{meta.get('version')}",
        "labels": etiquetas,
        "thresholds": umbrales["thresholds"],
        "featureDim": FEATURE_DIM,
        "faceDim": FACE_BLOCK_DIM,
        "nets": capas_json,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    peso = (out / "alfabeto.bin").stat().st_size
    print(f"{len(redes)} redes · {peso / 1e6:.1f} MB → {out}")

    # ── Vectores de prueba para TypeScript ──
    casos = casos_reales(args.casos_por_letra, etiquetas)
    vectores = []
    for c in casos:
        lm, world = np.array(c["lm"]), np.array(c["world"])
        ojo, boca = c.pop("_ref")
        cara = build_face_features(lm, ojo, boca, mirrored=c["mirrored"])
        lm_m, w_m = (mirror(lm), mirror(world)) if c["mirrored"] else (lm, world)
        feats = build_features(lm_m, w_m)
        x514 = np.concatenate([feats] + [cara] * args.face_repeat)[None].astype(np.float32)
        c["features"] = feats.round(6).tolist()
        c["face"] = cara.round(6).tolist()
        c["probs"] = modelo.predict(x514, verbose=0)[0].round(6).tolist()
        vectores.append(c)
    destino = args.web / "src" / "lib" / "alphabet-classifier.vectors.json"
    destino.write_text(json.dumps({"labels": etiquetas, "cases": vectores}, ensure_ascii=False),
                       encoding="utf-8")
    zurdas = sum(c["mirrored"] for c in vectores)
    print(f"{len(vectores)} vectores de prueba ({zurdas} manos izquierdas) → {destino}")


if __name__ == "__main__":
    main()
