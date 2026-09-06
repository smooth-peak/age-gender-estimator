import os

import numpy as np
import streamlit as st
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender.tflite")
IMG_SIZE = 128

st.set_page_config(page_title="Age & Gender Estimator", page_icon=":bust_in_silhouette:")


@st.cache_resource
def load_interpreter():
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp

interp = load_interpreter()
gender_out_idx = next(
    i for i, d in enumerate(interp.get_output_details()) if list(d["shape"]) == [1, 2]
)
age_out_idx = next(
    i for i, d in enumerate(interp.get_output_details()) if list(d["shape"]) == [1, 1]
)
input_idx = interp.get_input_details()[0]["index"]

st.title("Age & Gender Estimator")
st.write("Upload a face photo and get a predicted gender and approximate age.")

uploaded = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)[None, ...] / 127.5 - 1.0

    interp.set_tensor(input_idx, arr)
    interp.invoke()
    gender_probs = np.array(interp.get_tensor(interp.get_output_details()[gender_out_idx]["index"]))[0]
    age = float(interp.get_tensor(interp.get_output_details()[age_out_idx]["index"])[0][0])
    gender = "male" if np.argmax(gender_probs) == 0 else "female"

    uploaded.seek(0)
    st.image(uploaded, caption="Your photo", use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Gender", gender)
    c2.metric("Confidence", f"{float(gender_probs.max()):.2%}")
    c3.metric("Age", f"{int(round(age))}")