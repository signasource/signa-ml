"""
Pérdidas propias, registradas para que cualquier script pueda cargar el modelo.

Viven acá y no adentro de train_alphabet.py porque el .keras guarda una
referencia a la pérdida con la que se compiló cada red, y quien lo cargue
(export_alphabet_for_app.py, por ejemplo) tiene que poder resolverla.
"""
from __future__ import annotations

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="signa")
class SmoothedSparseCategoricalCrossentropy(keras.losses.Loss):
    """
    Entropía cruzada con label smoothing sobre etiquetas enteras.

    Keras 3 sacó `label_smoothing` de SparseCategoricalCrossentropy; la
    categórica sobre one-hot sí lo tiene.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.0, name="smoothed_sparse_cce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.smoothing = float(smoothing)

    def call(self, y_true, y_pred):
        y = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), self.num_classes)
        return keras.losses.categorical_crossentropy(y, y_pred, label_smoothing=self.smoothing)

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes, "smoothing": self.smoothing}


def _legacy_loss(y_true, y_pred):
    """
    Lo que referencia el signa_alphabet_v5.keras: se entrenó con la pérdida
    como función anónima llamada `loss`, que Keras no puede reconstruir. Para
    cargarlo sólo hace falta que el nombre resuelva; no se vuelve a entrenar.
    """
    raise RuntimeError("pérdida de entrenamiento: sólo para poder cargar el modelo")


# Pasar a load_model(custom_objects=...) para abrir modelos entrenados antes
# de que la pérdida fuera una clase registrada.
CUSTOM_OBJECTS = {"loss": _legacy_loss,
                  "SmoothedSparseCategoricalCrossentropy": SmoothedSparseCategoricalCrossentropy}
