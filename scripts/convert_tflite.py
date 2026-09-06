import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(BASE_DIR, "models", "age_gender.keras")
DST = os.path.join(BASE_DIR, "models", "age_gender.tflite")

model = tf.keras.models.load_model(SRC)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite = converter.convert()
with open(DST, "wb") as f:
    f.write(tflite)
print("saved", DST, len(tflite), "bytes")