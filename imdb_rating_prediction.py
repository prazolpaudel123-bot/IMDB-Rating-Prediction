import streamlit as st
import pickle
import numpy as np
import os
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="IMDb Rating Predictor",
    page_icon="⭐",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Resources (FIXED)
# -----------------------------
@st.cache_resource
def load_resources():
    model_path = r"L:\Softwares\VS Code\Files\RNN\imdb_rating_prediction\imdb_rating_simplernn.h5"
    tokenizer_path = r"L:\Softwares\VS Code\Files\RNN\imdb_rating_prediction\tokenizer_imdb_rating.pickle"
    max_length_path = r"L:\Softwares\VS Code\Files\RNN\imdb_rating_prediction\max_length_imdb_rating.pickle"

    missing = []
    for path in [model_path, tokenizer_path, max_length_path]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        st.error("❌ Missing files:")
        for m in missing:
            st.write(f"- `{m}`")
        st.info("Please put the model, tokenizer and max_length files in the same folder as app.py")
        st.stop()

    # Fixed loading to avoid the mse deserialization error
    model = load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(max_length_path, "rb") as f:
        max_length = pickle.load(f)

    return model, tokenizer, max_length

model, tokenizer, max_length = load_resources()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Prediction Function
# -----------------------------
def predict_rating(text: str):
    if not text.strip():
        return None

    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")

    pred = model.predict(padded, verbose=0)[0][0]
    pred = float(np.clip(pred, 1.0, 10.0))
    return pred

# -----------------------------
# UI
# -----------------------------
st.title("⭐ IMDb Rating Predictor")
st.markdown("Paste a movie or TV show **plot description** and the model will predict its **IMDb rating**.")

with st.sidebar:
    st.header("⚙️ Model Info")
    st.write("- Model: **SimpleRNN**")
    st.write(f"- Max sequence length: `{max_length}`")
    st.write(f"- Vocabulary size: `{len(tokenizer.word_index)}`")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("The model predicts a continuous IMDb rating (1–10) from the plot description.")
    st.markdown("---")
    st.markdown("Made with Streamlit + TensorFlow")

description = st.text_area(
    "Enter plot description:",
    height=180,
    placeholder="Example: A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine..."
)

st.markdown("#### 💡 Try these examples:")
col1, col2, col3 = st.columns(3)

example1 = "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine in order to secure his family's future."
example2 = "Decades after their 1984 All Valley Karate Tournament bout, a middle-aged Daniel LaRusso and Johnny Lawrence find themselves martial-arts rivals again."
example3 = "In this high-octane action thriller, a cross-country road trip becomes a highway to hell for Brenda and her family."

if col1.button("Breaking Bad"):
    description = example1
if col2.button("Cobra Kai"):
    description = example2
if col3.button("Action Movie"):
    description = example3

if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
    if not description.strip():
        st.warning("Please enter a plot description first.")
    else:
        with st.spinner("Predicting rating..."):
            predicted_rating = predict_rating(description)

        st.markdown("---")
        st.subheader("Predicted IMDb Rating")
        st.markdown(f"## ⭐ {predicted_rating:.2f} / 10")
        st.progress(predicted_rating / 10)

        if predicted_rating >= 8.5:
            st.success("Excellent! This sounds like a highly rated title.")
        elif predicted_rating >= 7.0:
            st.info("Good rating. This plot seems promising.")
        elif predicted_rating >= 5.5:
            st.warning("Average rating. Mixed expectations.")
        else:
            st.error("Low rating. The model thinks this plot may not be well received.")

        st.caption(f"Raw predicted score: `{predicted_rating:.4f}`")

st.markdown("---")
st.caption("IMDb Rating Prediction | SimpleRNN | Trained on plot descriptions")