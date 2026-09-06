# Age & Gender Estimator

Streamlit web app that predicts a person's gender and approximate age from a face photo.

- Model: MobileNetV2 (transfer learning) with dual heads trained on the UTKFace dataset.
- Run locally: `python -m streamlit run streamlit_app.py`

## Deploy

Hosted free on [Streamlit Community Cloud](https://streamlit.io/cloud) from this GitHub repository. Push to GitHub, then on streamlit.io click "Create app" and select this repo, branch, and `streamlit_app.py` as the main file.