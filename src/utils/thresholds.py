"""
Elección del umbral de decisión de una clase.

Vive acá y no al lado de un calibrador porque lo usan los dos —el de señas
dinámicas y el del abecedario— y tenerlo en uno obligaba al otro a importar un
script hermano, que es una dependencia que no debería existir entre dos
entradas de línea de comandos.
"""
from __future__ import annotations

import numpy as np

MIN_THRESHOLD = 0.08


def pick_threshold(p_letter, is_letter, min_precision, min_recall):
    """
    Umbral de aceptación para "esto es la letra L".

    Buscamos el mayor recall posible SIN que se caiga la precisión: que la app
    acepte la letra cuando la hacés bien, pero que no acepte cualquier cosa.
    Entre los umbrales con precisión aceptable elegimos el de mejor recall, y
    desempatamos por F1.

    Si ninguno llega al piso de precisión (pasa con letras muy solapadas con una
    vecina), caemos al de mejor F1. Esa letra va a ser permisiva, y el reporte
    la marca para que se sepa cuáles son.
    """
    candidates = np.unique(np.round(
        np.concatenate([p_letter, [MIN_THRESHOLD, 0.5, 0.99]]), 3))
    candidates = candidates[candidates >= MIN_THRESHOLD]

    scored = []
    for t in candidates:
        pred = p_letter >= t
        tp = float((pred & is_letter).sum())
        fp = float((pred & ~is_letter).sum())
        fn = float((~pred & is_letter).sum())
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        scored.append((f1, float(t), precision, recall))

    if not scored:
        return (0.0, 0.5, 0.0, 0.0)

    ok = [c for c in scored if c[2] >= min_precision]
    if ok:
        # El más permisivo que todavía sea preciso: en una app de práctica, que
        # no te acepte una letra bien hecha molesta más que aceptar alguna de
        # más. min_precision es el freno que evita irse al placebo.
        best = max(ok, key=lambda c: (round(c[3], 3), c[0]))
        if best[3] >= min_recall:
            # Si ya sobra recall, apretamos un poco y nos quedamos con el de
            # mejor F1 entre los que igual llegan al objetivo.
            good = [c for c in ok if c[3] >= min_recall]
            # Ante empate de F1 gana el umbral MÁS ALTO: mismo compromiso, pero
            # menos falsos positivos. Sin esto la elección quedaba en manos del
            # orden de la lista, y una seña como "hermano" caía en un umbral
            # bajo que disparaba con cualquier movimiento.
            return max(good, key=lambda c: (round(c[0], 3), c[1]))
        return best

    return max(scored, key=lambda c: c[0])


def center_threshold(p_letter, threshold):
    """
    Corre el umbral al medio del hueco donde cayó, sin cambiar ninguna decisión.

    `pick_threshold` devuelve un umbral que coincide con el puntaje de alguna
    foto: el de la última positiva que todavía acepta. Cualquier valor entre el
    puntaje de abajo más cercano y ese separa las fotos de calibración
    exactamente igual, pero quedarse en el borde no deja margen: en vivo las
    probabilidades salen promediadas entre frames y un poco más bajas que en una
    foto limpia. Con la A pasó eso: 15 fotos casi iguales, umbral 0.81 pegado a
    la A más floja, y una A de frente o a otra altura no llegaba nunca. El medio
    del hueco reparte el margen entre aceptar de menos y aceptar de más.
    """
    p = np.asarray(p_letter, dtype=np.float64)
    below = p[p < threshold]
    above = p[p >= threshold]
    if not len(below) or not len(above):
        return threshold
    return max(MIN_THRESHOLD, float((below.max() + above.min()) / 2))
