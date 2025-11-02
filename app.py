import streamlit as st
import numpy as np
import librosa
import joblib
import os
from pydub import AudioSegment
from audiorecorder import audiorecorder

# =============================
# Load Model
# =============================
model = joblib.load("model_randomforest.pkl")

st.set_page_config(page_title="Prediksi Suara Buka/Tutup", page_icon="🎤", layout="centered")

st.title("🎤 Aplikasi Prediksi Suara (Buka / Tutup)")
st.write("Unggah file suara atau rekam langsung untuk mengetahui hasil prediksi model.")
st.markdown("---")

# =============================
# PILIH OPSI INPUT
# =============================
option = st.radio("Pilih sumber audio:", ["🎙️ Rekam Suara", "📂 Upload File"])

file_path = None  # inisialisasi
audio_bytes = None

# --- Jika pilih rekam ---
if option == "🎙️ Rekam Suara":
    st.write("Tekan tombol di bawah ini untuk mulai/berhenti merekam:")
    audio = audiorecorder("Mulai / Berhenti Rekam", "🎙️ Rekam")

    if len(audio) > 0:
        # Simpan hasil rekaman ke file sementara
        file_path = "temp_rekam.wav"
        audio.export(file_path, format="wav")

        # Tampilkan player untuk mendengarkan hasil rekaman
        st.audio(file_path, format="audio/wav")

# --- Jika pilih upload ---
else:
    uploaded_file = st.file_uploader("Pilih file audio", type=["wav", "m4a", "mp3", "ogg"])
    if uploaded_file is not None:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(file_path)

# =============================
# PREDIKSI
# =============================
def predict_audio(file_path, model):
    try:
        # Gunakan pydub untuk semua format umum
        audio = AudioSegment.from_file(file_path)
        signal = np.array(audio.get_array_of_samples()).astype(np.float32)
        signal = signal / np.iinfo(audio.array_type).max
        sr = audio.frame_rate

        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        pred = model.predict(mfcc_mean)[0]
        return pred
    except Exception as e:
        return f"Error: {e}"

# --- Jalankan prediksi ---
if file_path is not None:
    st.markdown("---")
    if st.button("🔍 Prediksi Sekarang"):
        with st.spinner("Memproses audio..."):
            result = predict_audio(file_path, model)
        st.success(f"🎯 Hasil Prediksi: **{result.upper()}**")
