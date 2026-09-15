"""
Calibra un umbral por seña para el modo VERIFICACIÓN de las señas dinámicas.

Es el mismo razonamiento que scripts/calibrate_alphabet.py, pero sobre clips de
30 frames en vez de fotos: en un ejercicio la app ya sabe qué seña pidió, así
que no necesita resolver "¿cuál de las 13 es?" sino "¿esto es un 'mama'?". Un
umbral por seña importa porque no todas salen con la misma confianza: hay señas
que el modelo da al 99% y otras que reparten con una vecina parecida (mama/papa
se hacen casi en el mismo lugar) y nunca pasan un umbral global.

Cómo:
    1. Validación cruzada agrupada por clip original → probabilidades
       out-of-fold. Las aumentaciones de un clip viajan siempre con él: si
       quedaran repartidas entre train y test, el umbral saldría optimista.
    2. Para cada seña S, busca el umbral t_S con el mejor recall que todavía
       mantiene la precisión (misma función que el abecedario).
    3. A las señas que se separan solas les deja margen para el uso en vivo
       (ver CEDE_VIVO).
    4. Guarda los umbrales al lado del .tflite. export_for_app.py los copia al
       manifiesto que consume la app.

Uso:
    python scripts/calibrate_signs.py
    python scripts/calibrate_signs.py --folds 5 --min-recall 0.85
    python scripts/calibrate_signs.py --recalibrate   # sólo re-deriva umbrales
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.thresholds import pick_threshold
from scripts.train import build_lstm_model
from src.data.normalization import (con_encuadres, drop_pose, normalize_dataset,
                                    solo_con_manos)
from src.utils.labels import get_signs, load_config

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
EXPORTS_DIR = ROOT / "models" / "exports"
REPORTS_DIR = ROOT / "reports"

# La seña de reposo existe para que el modelo tenga dónde mandar "no está
# haciendo nada". Nunca se pide en un ejercicio, así que no lleva umbral, pero
# sigue contando como negativo de todas las demás.
REST_LABEL = "reposo"

# Margen para uso en vivo.
#
# El umbral sale de maximizar F1 sobre clips recortados a mano: los 30 frames
# son exactamente la seña, ni un frame de más. En la app la ventana corre sola y
# nunca cae tan bien — entra con algo de reposo adelante o atrás— así que la
# misma seña bien hecha da una probabilidad más baja que en el dataset. Un
# umbral pegado al óptimo de F1 no tiene de dónde ceder y la seña no entra nunca.
#
# Donde la seña se separa sola (el 10% peor de sus clips igual da MUY alta), se
# le cede esto. Donde el modelo ya duda, no se toca: ahí aflojar sería aceptar
# cualquier cosa.
#
# Se cede una franja fija y no una fracción del q10, que es lo que había antes.
# Con una fracción, cuanto mejor separaba el modelo una seña más barata salía:
# 'hermano' tiene q10 = 1.00 y terminaba en 0.60 — cuarenta puntos de regalo, y
# en el teléfono se disparaba con cualquier movimiento. La franja fija no
# depende de lo seguro que esté el modelo, que es de lo que no tiene que
# depender.
CEDE_VIVO = 0.10
SEGURA_DESDE = 0.8

# Cuánto recall se acepta resignar a cambio de precisión.
#
# El umbral que maximiza F1 puede quedar en cualquier punto de una zona plana:
# 'hermano' da el mismo F1 de 0.35 a 0.60 y el óptimo cae en el piso, que en el
# teléfono es "se dispara con cualquier cosa". Después de elegirlo se prueba
# subirlo mientras el recall no baje más que esto.
#
# El valor tiene que ser chico: con una tolerancia grande, una seña floja como
# 'mama' —que pierde 30 puntos de recall entre 0.35 y 0.80— terminaría con un
# umbral inalcanzable. Con 0.02 se mueve sólo donde la curva es plana.
TOLERANCIA_RECALL = 0.02

# Piso absoluto del umbral.
#
# El del abecedario es 0.08, y para fotos tiene sentido. Acá no: la ventana
# corre sola sobre video y la mayor parte del tiempo contiene a alguien parado
# con las manos arriba, que no es ninguna de las doce señas pero tampoco es un
# negativo que el calibrador haya visto — sus negativos son clips limpios de
# las otras señas. Con umbrales de 0.08 o 0.22 el ejercicio aceptaba una
# postura quieta. Por debajo de esto no se baja, por más que los datos lo
# permitan.
PISO_VIVO = 0.35

# sequence_12_aug0.npy → clip 12. El grupo es el clip, no el archivo.
CLIP_RE = re.compile(r"sequence_(\d+)")


def load_grouped(config: dict):
    """
    Igual que train.load_dataset, pero además devuelve de qué clip salió cada
    secuencia y si es una aumentación.

    Returns:
        X (N,T,258), y (N,), grupos (N,) clave "<seña>:<clip>", aug (N,) 0/1,
        etiquetas [labels en orden de clase]
    """
    largo = config["training"]["sequence_length"]
    X, y, grupos, aug, etiquetas = [], [], [], [], []

    for sign in get_signs():
        carpeta = DATA_DIR / sign["folder"]
        secuencias = sorted(carpeta.glob("sequence_*.npy")) if carpeta.exists() else []
        if not secuencias:
            print(f"  [SKIP] sin datos para '{sign['label']}'")
            continue

        clase = len(etiquetas)
        etiquetas.append(sign["label"])
        cargadas = 0

        for ruta in secuencias:
            seq = np.load(ruta)
            if seq.shape != (largo, 258):
                continue
            m = CLIP_RE.match(ruta.stem)
            util = solo_con_manos(seq)
            if util is None:
                continue
            X.append(util)
            y.append(clase)
            grupos.append(f"{sign['label']}:{m.group(1) if m else ruta.stem}")
            aug.append(1 if "_aug" in ruta.stem else 0)
            cargadas += 1

        print(f"  ✓ {sign['label']}: {cargadas} secuencias (clase {clase})")

    if len(etiquetas) < 2:
        print("ERROR: hacen falta al menos dos señas con datos.")
        sys.exit(1)

    return (np.array(X), np.array(y), np.array(grupos),
            np.array(aug), etiquetas)


def entrenar(X_tr, y_tr, X_val, y_val, n_clases, cfg, epochs):
    import tensorflow as tf

    modelo = build_lstm_model(cfg["sequence_length"], 258, n_clases)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.fit(
        X_tr, tf.keras.utils.to_categorical(y_tr, n_clases),
        validation_data=(X_val, tf.keras.utils.to_categorical(y_val, n_clases)),
        epochs=epochs, batch_size=cfg["batch_size"], verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True)],
    )
    return modelo


def out_of_fold_probs(X, y, grupos, aug, etiquetas, cfg, epochs, folds, seed):
    """
    Probabilidades para cada clip apartado, puntuado por un modelo que no lo vio.

    Cada clip apartado se puntúa en varios encuadres, no sólo en el que se
    grabó. El umbral que sale de acá es el que va a usar la app, y la app nunca
    ve una seña perfectamente encuadrada: su ventana son los últimos 2,5 s de
    video, con la seña entrando o saliendo. Calibrar sólo sobre clips centrados
    daba umbrales que en el teléfono no se alcanzaban nunca.

    Devuelve (P, Y) ya apiladas: una fila por clip y encuadre evaluado.
    """
    unicos = np.array(sorted(set(grupos.tolist())))
    partes = np.array_split(np.random.default_rng(seed).permutation(unicos),
                            min(folds, len(unicos)))
    rep = etiquetas.index(REST_LABEL) if REST_LABEL in etiquetas else -1

    P, Y = [], []

    for k, apartados in enumerate(partes, 1):
        fuera = np.isin(grupos, apartados)
        train = ~fuera
        test = fuera & (aug == 0)
        if test.sum() == 0 or len(set(y[train].tolist())) < len(etiquetas):
            print(f"  fold {k}: salteado")
            continue

        idx = np.flatnonzero(train)
        rng = np.random.default_rng(seed + k)
        val = rng.choice(idx, size=max(1, len(idx) // 10), replace=False)
        solo_train = train.copy()
        solo_train[val] = False

        # El relleno de reposo sale siempre del lado de entrenamiento, tanto
        # para aumentar como para evaluar: los clips apartados no aportan ni
        # un frame a nada que el modelo haya visto.
        reposos = X[solo_train & (y == rep)] if rep >= 0 else X[:0]
        X_tr, y_tr = con_encuadres(X[solo_train], y[solo_train], reposos,
                                   np.random.default_rng(seed + k))

        modelo = entrenar(X_tr, y_tr, X[val], y[val],
                          len(etiquetas), cfg, epochs)

        X_te, y_te = con_encuadres(X[test], y[test], reposos,
                                   np.random.default_rng(seed + 100 + k))
        p_te = modelo.predict(X_te, verbose=0)
        P.append(p_te)
        Y.append(y_te)
        print(f"  fold {k}: {test.sum()} clips × {len(X_te) // max(test.sum(), 1)} "
              f"encuadres → id {float((p_te.argmax(1) == y_te).mean()):.1%}")

    return np.concatenate(P), np.concatenate(Y)


def metricas(p, es, t):
    """F1, precisión y recall de "esto es la seña" en un umbral dado."""
    pred = p >= t
    tp = float((pred & es).sum())
    fp = float((pred & ~es).sum())
    fn = float((~pred & es).sum())
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall), precision, recall


def apretar(p, es, thr, tolerancia):
    """El umbral más alto que no resigna más recall que `tolerancia`."""
    base = metricas(p, es, thr)[2]
    mejor = thr
    for t in np.unique(np.round(p, 2)):
        if t > thr and metricas(p, es, float(t))[2] >= base - tolerancia:
            mejor = float(t)
    return mejor


def main() -> None:
    p = argparse.ArgumentParser(description="Calibra umbrales por seña dinámica.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    # 0.70 y no 0.90: con 0.90 el umbral de "mama" se iba a 0.96 y su recall a
    # 32%, o sea inusable. La precisión de acá mide cuántos clips de OTRAS señas
    # superan el umbral, pero en el ejercicio la persona está intentando
    # justamente la seña pedida, y además hay tres filtros más antes de
    # confirmar: que haya manos, que haya movimiento, y sostenerla 700 ms. Ahí
    # el costo de un umbral alto —no aceptar nunca una seña bien hecha— pesa
    # mucho más que el de uno bajo. Medido: bajar de 0.70 no mejora nada.
    p.add_argument("--min-precision", type=float, default=0.70,
                   help="Precisión mínima por seña: evita umbrales que acepten cualquier cosa")
    p.add_argument("--min-recall", type=float, default=0.85,
                   help="Recall objetivo por seña (no se fuerza si rompe la precisión)")
    p.add_argument("--cache", type=Path, default=REPORTS_DIR / "signs_oof_probs.npz")
    p.add_argument("--recalibrate", action="store_true",
                   help="Reusar probabilidades cacheadas y sólo recalcular umbrales")
    p.add_argument("--live-margin", type=float, default=CEDE_VIVO,
                   help="Cuánta probabilidad se le cede a las señas que se separan solas")
    p.add_argument("--tolerancia", type=float, default=TOLERANCIA_RECALL,
                   help="Recall que se acepta perder a cambio de precisión")
    p.add_argument("--piso", type=float, default=PISO_VIVO,
                   help="Umbral mínimo por seña, pase lo que pase")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    config = load_config()
    cfg = config["training"]
    epochs = args.epochs or cfg["epochs"]

    print("\nCargando datos...")
    X, y, grupos, aug, etiquetas = load_grouped(config)
    print(f"\n{len(X)} secuencias · {len(etiquetas)} señas · "
          f"{len(set(grupos.tolist()))} clips originales")

    # Mismo trato que en el entrenamiento: sin la pose como evidencia.
    X = drop_pose(normalize_dataset(X))

    # La validación cruzada tarda varios minutos y los umbrales se re-derivan en
    # milisegundos: cacheamos las probabilidades para poder mover los pisos sin
    # volver a entrenar cinco modelos.
    if args.recalibrate and args.cache.exists():
        cacheado = np.load(args.cache, allow_pickle=True)
        P, Y = cacheado["probs"], cacheado["y"]
        print(f"Probabilidades cacheadas: {args.cache}")
    else:
        print("\nProbabilidades out-of-fold:")
        P, Y = out_of_fold_probs(X, y, grupos, aug, etiquetas, cfg, epochs,
                                 args.folds, args.seed)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, probs=P, y=Y)
        print(f"  cacheadas en {args.cache} (--recalibrate las reusa)")

    umbrales, filas = {}, []
    for i, seña in enumerate(etiquetas):
        if seña == REST_LABEL:
            continue
        es = Y == i
        if es.sum() == 0:
            umbrales[seña] = 0.5
            continue
        f1, thr, precision, recall = pick_threshold(
            P[:, i], es, args.min_precision, args.min_recall)
        q10 = float(np.quantile(P[es, i], 0.10))
        if q10 >= SEGURA_DESDE:
            thr = min(thr, q10 - args.live_margin)
        thr = max(thr, args.piso)
        thr = apretar(P[:, i], es, thr, args.tolerancia)
        # Recalculados en el umbral que realmente queda: los que devuelve
        # pick_threshold son los de su propia elección, antes del piso.
        f1, precision, recall = metricas(P[:, i], es, thr)
        umbrales[seña] = round(thr, 3)
        filas.append((seña, thr, recall, precision, f1,
                      float((P[es].argmax(1) == i).mean()), int(es.sum())))

    filas.sort(key=lambda r: r[2])
    lineas = [
        "Umbrales de verificación por seña (out-of-fold, agrupado por clip)",
        f"Identificación global: {float((P.argmax(1) == Y).mean()):.1%}",
        "",
        "seña              umbral  recall  precisión   F1   ident.  n",
    ]
    for seña, thr, rec, prec, f1, id_acc, n in filas:
        lineas.append(f"  {seña:<16} {thr:5.2f}   {rec:5.0%}    {prec:5.0%}   "
                      f"{f1:.2f}  {id_acc:5.0%}  {n:>3}")

    permisivas = [r[0] for r in filas if r[3] < args.min_precision]
    if permisivas:
        lineas += ["", f"Señas permisivas (precisión < {args.min_precision:.0%}): "
                   + ", ".join(permisivas)
                   + " — se aceptan fácil, pero también aceptan vecinas parecidas."]
    lineas += [
        "",
        f"Recall más bajo entre las {len(filas)} señas: {min(r[2] for r in filas):.0%}",
        "",
        "Cómo leerlo:",
        "  recall = de las veces que hacés la seña, cuántas la acepta el ejercicio.",
        "  ident. = cuántas veces gana entre todas sin saber qué se pidió.",
        f"  '{REST_LABEL}' no lleva umbral: nunca se pide, sólo hace de negativo.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reporte = REPORTS_DIR / "signs_thresholds.txt"
    reporte.write_text("\n".join(lineas), encoding="utf-8")
    print("\n" + "\n".join(lineas))

    modelos = sorted(EXPORTS_DIR.glob("signa_model_v*.tflite"),
                     key=lambda m: int("".join(c for c in m.stem.split("_v")[-1]
                                               if c.isdigit()) or 0))
    if modelos:
        salida = modelos[-1].with_name(modelos[-1].stem + "_thresholds.json")
        salida.write_text(json.dumps({
            "thresholds": umbrales,
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            "identification_accuracy": round(float((P.argmax(1) == Y).mean()), 4),
            "clips_scored": int(len(Y)),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nUmbrales: {salida}")
    print(f"Reporte:  {reporte}")


if __name__ == "__main__":
    main()
