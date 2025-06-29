import os
import gdown
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# URL Google Drive file best model aspek petugas kamu (ganti ini dengan link asli)
MODEL_FILES = {
    "config.json": "https://drive.google.com/uc?id=ID_FILE_CONFIG",
    "model.safetensors": "https://drive.google.com/uc?id=ID_FILE_MODEL",
    "tokenizer.json": "https://drive.google.com/uc?id=ID_FILE_TOKENIZER",
    "vocab.txt": "https://drive.google.com/uc?id=ID_FILE_VOCAB",
    "special_tokens_map.json": "https://drive.google.com/uc?id=ID_FILE_SPECIAL_TOKENS"
}

MODEL_FOLDER = "petugas_model"

def download_model():
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    for filename, url in MODEL_FILES.items():
        path = os.path.join(MODEL_FOLDER, filename)
        if not os.path.exists(path):
            with st.spinner(f"Mengunduh {filename}..."):
                gdown.download(url, path, quiet=False)

@st.cache_resource(show_spinner=True)
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_FOLDER)

@st.cache_resource(show_spinner=True)
def load_model():
    config = AutoConfig.from_pretrained(MODEL_FOLDER)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER, config=config)
    model.eval()
    return model

def main():
    st.title("📌 Prediksi Aspek: Petugas Haji")

    download_model()

    tokenizer = load_tokenizer()
    model = load_model()

    text = st.text_area("Masukkan teks:", height=150)

    if st.button("Cek Aspek"):
        if not text.strip():
            st.warning("Masukkan teks dulu ya!")
            return

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

        if pred == 1:
            st.success("✅ Teks ini **termasuk aspek petugas**.")

            st.markdown("---")
            st.markdown("🔄 Lanjutkan prediksi sentimen:")
            st.markdown(
                '[Klik di sini untuk buka aplikasi prediksi sentimen aspek petugas](https://sentimen-petugas.streamlit.app)',
                unsafe_allow_html=True
            )
        else:
            st.warning("⛔ Teks ini **tidak termasuk aspek petugas**.")

if __name__ == "__main__":
    main()
