import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender.keras")
IMG_SIZE = 128

st.set_page_config(page_title="Age & Gender Estimator", page_icon=":bust_in_silhouette:")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("Age & Gender Estimator")
st.write("Upload a face photo and get a predicted gender and approximate age.")

uploaded = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    bytes_data = uploaded.getvalue()
    tmp = os.path.join(BASE_DIR, "models", "_tmp_upload.jpg")
    with open(tmp, "wb") as f:
        f.write(bytes_data)

    img = tf.keras.utils.load_img(tmp, target_size=(IMG_SIZE, IMG_SIZE))
    arr = preprocess_input(tf.keras.utils.img_to_array(img)[None, ...])

    out = model.predict(arr, verbose=0)
    if isinstance(out, dict):
        gender_probs = out["gender"][0]
        age = float(out["age"][0][0])
    else:
        gender_probs = out[0][0]
        age = float(out[1][0][0])
    gender = "male" if np.argmax(gender_probs) == 0 else "female"

    st.image(uploaded, caption="Your photo", use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Gender", gender)
    c2.metric("Confidence", f"{float(gender_probs.max()):.2%}")
    c3.metric("Age", f"{int(round(age))}")