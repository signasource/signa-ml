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
notebooks/      Exploración y experimentos
scripts/        CLI para grabar datos, extraer landmarks, entrenar, exportar
src/            Código reutilizable (extractor, modelo, inferencia)
models/exports/ Artefactos .tflite listos para integrar en signa-mobile
configs/        Hiperparámetros y lista de señas objetivo
```

---

## Flujo de trabajo

```
1. Grabar señas        →  scripts/collect_data.py
2. Extraer landmarks   →  scripts/extract_landmarks.py
3. Entrenar            →  scripts/train.py          (próximamente)
4. Exportar .tflite    →  scripts/export_tflite.py  (próximamente)
```

Para explorar MediaPipe antes de grabar datos, abrí el notebook:

```bash
jupyter notebook notebooks/01_mediapipe_exploration.ipynb
```

---

## Estado actual

- [x] Estructura base del repo
- [x] Exploración de MediaPipe (pose + manos)
- [x] Captura de datos desde cámara (`collect_data.py`)
- [x] Extracción de landmarks (`extract_landmarks.py`)
- [ ] Entrenamiento del clasificador
- [ ] Exportación a TensorFlow Lite
- [ ] Integración con signa-mobile

---

## Artefacto de salida

`models/exports/signa_model_vX.tflite` — este archivo es el que se integra en la app Flutter.

---

## Contribuir

Las señas objetivo y sus IDs están en `configs/signs_config.yaml`.  
Para agregar una seña nueva: agregala al config, grabá al menos 30 repeticiones con `collect_data.py`, y re-corré el pipeline completo.
