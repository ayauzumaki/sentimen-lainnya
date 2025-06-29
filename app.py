import os
import gdown
import streamlit as st
import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# Link Google Drive ke file model
MODEL_FILES = {
    "config.json": "https://drive.google.com/uc?id=1-gaeEZS51znvhsvSdnez9XB-xKvR2Dih",
    "best_model.pt": "https://drive.google.com/uc?id=1USuLqLopkGwJY6EtBTARdor6qvHcowW_",
    "vocab.txt": "https://drive.google.com/uc?id=1Ur8adye08EcCoQ74YMIgFCPJenkdtqhA",
    "special_tokens_map.json": "https://drive.google.com/uc?id=1-lurkvcFx02DmjMqzIc9Z4LGRGRQ9AuS",
    "tokenizer_config.json": "https://drive.google.com/uc?id=1-tHk4S9UMk3xdosTpkgeCkxsP3JWB2xJ"
}

MODEL_FOLDER = "petugas_model"

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

@st.cache_resource(show_spinner=True)
def load_tokenizer():
    return BertTokenizer.from_pretrained(MODEL_FOLDER)

@st.cache_resource(show_spinner=True)
def load_model():
    config = BertConfig.from_pretrained(MODEL_FOLDER)
    model = BertForSequenceClassification.from_pretrained(MODEL_FOLDER, config=config)
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
