import os
import re
import string
import gdown
import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

MODEL_FILES = {
    "config.json": "https://drive.google.com/uc?id=1PwLc9C2ifJXqYa2Dq41aWTE7B6o-Yew_",
    "best_model.pt": "https://drive.google.com/uc?id=1a2n4epYb1y-XKRy4AJ13M5XAew-U-rND",
    "vocab.txt": "https://drive.google.com/uc?id=1eWl-oz6dlaugnqaj_bZkQ_bS9UjgmDih",
    "special_tokens_map.json": "https://drive.google.com/uc?id=11Txk9jHUD-ZJpt1VP0RP1tthU0PJHS_Sm",
    "tokenizer_config.json": "https://drive.google.com/uc?id=1lltqIoi41mLXg5q-yO2dRmf90FVeoRBI"
}
MODEL_FOLDER = "sentimen_lainnya_model"

KAMUS_CSV_URL = "https://drive.google.com/uc?id=1fGWZu5qVYJa-pv078spaLE4urs5zDDPV"
KAMUS_PATH = "kamus.csv"

def download_model():
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    for filename, url in MODEL_FILES.items():
        path = os.path.join(MODEL_FOLDER, filename)
        if filename == "best_model.pt":
            path = os.path.join(MODEL_FOLDER, "pytorch_model.bin")
        if not os.path.exists(path):
            with st.spinner(f"Mengunduh {filename}..."):
                gdown.download(url, path, quiet=False)

def download_kamus():
    if not os.path.exists(KAMUS_PATH):
        with st.spinner("Mengunduh kamus slang..."):
            gdown.download(KAMUS_CSV_URL, KAMUS_PATH, quiet=False)

@st.cache_resource(show_spinner=True)
def load_tokenizer():
    return BertTokenizer.from_pretrained(MODEL_FOLDER)

@st.cache_resource(show_spinner=True)
def load_model():
    config = BertConfig.from_pretrained(MODEL_FOLDER)
    model = BertForSequenceClassification.from_pretrained(MODEL_FOLDER, config=config)
    model.eval()
    return model

@st.cache_resource
def load_kamus():
    df = pd.read_csv(KAMUS_PATH)
    return dict(zip(df['slang'], df['formal']))

def preprocess(text, kamus_slang):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([kamus_slang.get(word, word) for word in text.split()])
    return text.strip()

def main():
    st.title("🧭 Prediksi Sentimen - Aspek Lainnya Haji")

    download_model()
    download_kamus()

    tokenizer = load_tokenizer()
    model = load_model()
    kamus_slang = load_kamus()

    text = st.text_area("Masukkan teks:", height=150)

    if st.button("Prediksi Sentimen"):
        if not text.strip():
            st.warning("Masukkan teks dulu ya!")
            return

        cleaned_text = preprocess(text, kamus_slang)
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

        if pred == 2:
            st.success("✅ Sentimen: **Positif**")
        else:  # pred == 0
            st.error("❌ Sentimen: **Negatif**")

if __name__ == "__main__":
    main()
