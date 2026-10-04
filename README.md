# signa-ml

Módulo de machine learning para reconocimiento de señas del Lenguaje de Señas Argentino (LSA).  
Parte del ecosistema [Signa](https://github.com/signasource).

El pipeline toma video de cámara → extrae landmarks con MediaPipe → entrena un clasificador → exporta un archivo `.tflite` que consume la app móvil.

---

## Stack

- Python 3.10+
- [MediaPipe](https://mediapipe.dev/) — detección de pose y manos
- TensorFlow / Keras — entrenamiento
- TensorFlow Lite — exportación para móvil
- OpenCV — captura de video

---

## Setup

```bash
git clone https://github.com/tu-org/signa-ml.git
cd signa-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Estructura

```
data/           Videos e imágenes crudas + landmarks extraídos (no versionado)
demo/           Escenas web para grabar el video de la demo
notebooks/      Exploración y experimentos
scripts/        CLI para grabar datos, extraer landmarks, entrenar, exportar
src/            Código reutilizable (extractor, modelo, inferencia)
models/exports/ Artefactos .tflite listos para integrar en signa-mobile
configs/        Hiperparámetros y lista de señas objetivo
```

---

## Flujo de trabajo

```
1. Grabar señas     →  scripts/collect_data.py       (captura y extrae en un paso)
2. Aumentar         →  scripts/augment_data.py
3. Entrenar         →  scripts/train.py --model lstm
4. Calibrar umbral  →  scripts/calibrate_signs.py
5. Llevar a la app  →  scripts/export_for_app.py
```

`collect_data.py` graba y extrae landmarks en la misma pasada, así que no hay
paso de extracción aparte. `extract_landmarks.py` sigue estando para procesar
videos que ya estén en `data/raw/`.

La calibración no es opcional: el modelo se usa en modo **verificación** —"¿esto
es *mama*?"— y cada seña necesita su propio umbral. Uno global deja afuera a las
señas que reparten probabilidad con una parecida.

Para explorar MediaPipe antes de grabar datos, abrí el notebook:

```bash
jupyter notebook notebooks/01_mediapipe_exploration.ipynb
```

---

## Estado actual

- [x] Estructura base del repo
- [x] Exploración de MediaPipe (pose + manos)
- [x] Captura de datos desde cámara (`collect_data.py`)
- [x] Extracción de landmarks (dentro de `collect_data.py`)
- [x] Entrenamiento del clasificador (señas dinámicas)
- [x] Exportación a TensorFlow Lite
- [x] Pipeline de abecedario / señas estáticas
- [x] Integración con signa-mobile (el .tflite viaja dentro de la app)

---

## Abecedario (señas estáticas)

Pipeline **aparte** del de señas dinámicas. Las señas del abecedario son formas
de mano fijas, así que en vez de secuencias de 30 frames con pose + manos, se
clasifica **una sola foto** usando sólo landmarks de mano.

| | señas dinámicas | abecedario |
|---|---|---|
| config | `configs/signs_config.yaml` | `configs/alphabet_config.yaml` |
| detector | MediaPipe Tasks: Pose + Hands | MediaPipe Tasks: Hands |
| entrada | (30, 258) secuencia | (258,) un frame |
| features | landmarks crudos normalizados por hombros | ver `src/data/hand_features.py` |
| modelo | dense / LSTM | MLP |
| export | `signa_model_vX.tflite` | `signa_alphabet_vX.tflite` |

### Datos

Las fotos van en `data/raw/abc/` (cualquier subcarpeta, se busca recursivo) con
la convención `LSA_<LETRA>_<FUENTE>.png`:

```
data/raw/abc/
├── Fotos/          LSA_A_001.png, LSA_B_001.png, ...
├── fotos_palo/     LSA_C_005.jpeg, ...
└── webcam_mateo/   LSA_A_100.png, ...   (generadas por collect_alphabet.py)
```

`<FUENTE>` es el **id del video del que se sacó el screenshot** (ver el Tracker
en Drive), no un índice de muestra. Es el dato más importante del nombre: el
split de entrenamiento agrupa por fuente, así el test set nunca comparte
señante con el train set.

### Flujo

```bash
# 0. Levantar las escenas para grabar el video (ver demo/README.md)
python demo/server.py --signs   # → http://localhost:8000/ (menú: nombre y señas)

# 1. Extraer landmarks de las fotos (con aumentación a nivel imagen)
python scripts/build_alphabet_dataset.py

# 2. Entrenar + exportar .tflite (con la posición respecto de la cara)
python scripts/train_alphabet.py --with-face --cv   # --cv = validación cruzada por fuente

# 3. Calibrar un umbral por letra (modo verificación)
python scripts/calibrate_alphabet.py --with-face

# 4. Llevarlo a la app (assets/ y el módulo nativo, golden incluido)
python scripts/export_alphabet_for_app.py

# 5. Llevarlo a la landing (signa-web: demo "Tu cámara te corrige", sin TFLite)
python scripts/export_alphabet_for_web.py

# Probar en vivo
python scripts/predict_alphabet_realtime.py

# (opcional pero muy recomendado) sumar fotos propias con tu webcam
python scripts/collect_alphabet.py --profile mateo
```

Salidas:

- `models/exports/signa_alphabet_vX.tflite` + `_meta.json` (labels, accuracy, CV)
- `reports/alphabet_extraction.csv` — una fila por foto, con score y si se
  detectó la mano
- `reports/alphabet_report.txt` — precisión por letra y las confusiones más
  frecuentes
- `reports/alphabet_confusion.png` — matriz de confusión

### Sobre la accuracy

`train_alphabet.py` reporta por defecto el split **por fuente**: entrena con
unos videos y testea con otros. Da un número más bajo que un split aleatorio,
pero es el que predice cómo va a andar frente a una cámara nueva. El split
`--split random` existe sólo como sanity check y su accuracy está inflada
(variantes aumentadas de la misma foto caen a los dos lados del split).

El modelo exportado es un **ensemble**: N redes con semillas distintas,
promediadas dentro de un único `.tflite` (misma interfaz, 258 entradas →
N_letras salidas). Se controla con `training.ensemble` en el config o
`--ensemble N`. Con 1 red el modelo pesa ~1,2 MB; con 5, ~6 MB.

### Límites conocidos

Varias letras del LSA se distinguen por **movimiento**, no por la forma de la
mano. Desde una sola foto son literalmente el mismo handshape y ningún modelo
estático las puede separar. En los datos actuales el par más afectado es
**N / Ñ**; le siguen **I / T**, **E / C** y **W / V / U**. `reports/alphabet_report.txt`
lista las confusiones ordenadas después de cada entrenamiento — conviene mirarlo
antes de elegir qué letras mostrar en una demo.

En vivo el reconocimiento es mejor que lo que sugiere la accuracy por foto:
`AlphabetRecognizer` promedia las probabilidades de varios frames, así que
acumula evidencia en vez de decidir con una sola imagen.

---

## Artefacto de salida

`models/exports/signa_model_vX.tflite` — este archivo es el que se integra en la app Flutter.

---

## Contribuir

Las señas objetivo y sus IDs están en `configs/signs_config.yaml`.  
Para agregar una seña nueva: agregala al config, grabá al menos 30 repeticiones con `collect_data.py`, y re-corré el pipeline completo.

## Scripts importantes

# procesar key points de señas en tiempo real por cámara

python scripts/collect_data.py --sign seña --sequences 15

# entrenar modelo

python scripts/train.py --model dense --epochs 50   

# predecir en tiempo real

python scripts/predict_realtime.py                    