import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

def read_csv(name):
    paths, ages, genders = [], [], []
    with open(os.path.join(BASE_DIR, "data", "prepared", name)) as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            ages.append(float(row["age"]))
            genders.append(int(row["gender"]))
    return paths, np.array(ages), np.array(genders)

train_paths, train_ages, train_genders = read_csv("train.csv")
val_paths, val_ages, val_genders = read_csv("val.csv")
print("train", len(train_paths), "val", len(val_paths))

def load_arrays(paths, ages, genders):
    x = np.zeros((len(paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    yg = np.zeros((len(paths), 2), dtype=np.float32)
    for i, p in enumerate(paths):
        img = tf.keras.utils.load_img(p, target_size=(IMG_SIZE, IMG_SIZE))
        x[i] = preprocess_input(tf.keras.utils.img_to_array(img))
        yg[i, genders[i]] = 1.0
    return x, {"gender": yg, "age": ages.astype(np.float32)}

x_train, y_train = load_arrays(train_paths, train_ages, train_genders)
x_val, y_val = load_arrays(val_paths, val_ages, val_genders)
print("arrays", x_train.shape, x_val.shape)

base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.2)(x)
gender_out = layers.Dense(2, activation="softmax", name="gender")(x)
age_out = layers.Dense(1, activation="relu", name="age")(x)

model = models.Model(base.input, {"gender": gender_out, "age": age_out})
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss={"gender": "categorical_crossentropy", "age": "mse"},
    loss_weights={"gender": 1.0, "age": 0.05},
    metrics={"gender": "accuracy", "age": "mae"},
)
model.summary()

hist = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(BASE_DIR, "models", "checkpoint.keras"),
            monitor="val_age_mae",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ],
)

model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "age_gender.keras"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist.history["gender_accuracy"], label="train")
axes[0].plot(hist.history["val_gender_accuracy"], label="val")
axes[0].set_title("Gender accuracy")
axes[0].legend()
axes[1].plot(hist.history["age_mae"], label="train")
axes[1].plot(hist.history["val_age_mae"], label="val")
axes[1].set_title("Age MAE (years)")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "training_curves.png"))
print("saved model + curves to", model_dir)