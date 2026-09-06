import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender.keras")
IMG_SIZE = 128


def main(path):
    if not os.path.exists(path):
        sys.exit("image not found: " + path)
    model = tf.keras.models.load_model(MODEL_PATH)
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = preprocess_input(tf.keras.utils.img_to_array(img)[None, ...])

    out = model.predict(arr, verbose=0)
    if isinstance(out, dict):
        gender_probs = out["gender"][0]
        age = float(out["age"][0][0])
    else:
        gender_probs = out[0][0]
        age = float(out[1][0][0])
    gender = "male" if np.argmax(gender_probs) == 0 else "female"

    print(f"photo:        {path}")
    print(f"gender:       {gender}  (confidence {float(gender_probs.max()):.2%})")
    print(f"age:          {int(round(age))} years")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: predict.py path/to/photo.jpg")
    main(sys.argv[1])