import streamlit as st
import time
from train import train_model
from generate import generate_music
from quality_testing import quality_test
import matplotlib.pyplot as plt
import base64
import os

# Page Config
st.set_page_config(page_title="🎵 AI Music Composer", layout="wide")

# ---------------- BACKGROUND + ANIMATION ----------------
st.markdown("""
<style>
body {
    background-image: url('C://Users//mkava//Desktop//AI-Music Composer//images//image-4.jpeg');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    font-family: 'Poppins';
}
.block { background: rgba(0,0,0,0.6); padding:25px; border-radius:15px; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "🎤 Training", "🎼 Generate", "📊 Quality Test"])

# --------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center;'>🎵 AI Music Composer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:20px;'>Train AI, generate music, and evaluate composition quality — all in one studio.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='block'>
    <h3>✨ Features</h3>
    • Train a custom Transformer model on your MIDI dataset <br>
    • Generate brand new melodies automatically <br>
    • Listen instantly with built-in audio player <br>
    • Run quality testing with visual graphs <br>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------------
# TRAINING
# --------------------------------------------------------
elif page == "🎤 Training":
    st.markdown("<h2>🎤 Train Your AI Model</h2>", unsafe_allow_html=True)
    st.markdown("<div class='block'>🧠 Configure your training below</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", 1, 25, 5)
    batch = col2.slider("Batch Size", 4, 64, 16)
    seq_len = col3.slider("Sequence Length", 20, 120, 50)

    progress = st.progress(0)
    status = st.empty()

    if st.button("🚀 Start Training"):
        with st.spinner("Training AI Model..."):
            try:
                checkpoint = train_model(epochs=epochs, batch_size=batch, seq_len=seq_len)
                progress.progress(100)
                status.success(f"🎉 Training complete! Checkpoint saved: {checkpoint}")
            except Exception as e:
                status.error(f"❌ Error: {e}")

# --------------------------------------------------------
# GENERATE
# --------------------------------------------------------
elif page == "🎼 Generate":
    st.markdown("<h2>🎼 Generate New Music</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    seed_note = col1.number_input("Seed note", min_value=20, max_value=90, value=60)
    seq = col2.slider("Sequence Length", 20, 150, 50)
    temp = col3.slider("Temperature", 0.5, 1.5, 1.0)

    if st.button("🎹 Generate Music Now"):
        with st.spinner("Generating melody... 🎶"):
            midi_file, wav_file = generate_music(seed_note=seed_note, seq_len=seq, temperature=temp)

        st.success("🎉 Music Generated Successfully!")
        st.audio(wav_file)

        with open(midi_file, "rb") as f:
            st.download_button("⬇ Download MIDI", f, file_name=os.path.basename(midi_file))

        with open(wav_file, "rb") as f:
            st.download_button("⬇ Download WAV", f, file_name=os.path.basename(wav_file))

# --------------------------------------------------------
# QUALITY TEST
# --------------------------------------------------------
elif page == "📊 Quality Test":
    st.markdown("<h2>📊 Quality Evaluation</h2>", unsafe_allow_html=True)
    st.markdown("<div class='block'>Checks pitch contour, audio brightness and generates score</div>", unsafe_allow_html=True)

    if st.button("🔍 Run Quality Test"):
        with st.spinner("Analyzing composition..."):
            score = quality_test()

        st.success(f"🏆 Quality Score: {score}/10")

        # Show graphs
        qual_dir = os.path.join(os.path.dirname(__file__), "quality_tests")
        pc = os.path.join(qual_dir, "pitch_contour.png")
        sc = os.path.join(qual_dir, "spectral_centroid.png")

        img1, img2 = st.columns(2)
        img1.image(pc, caption="MIDI Pitch Contour")
        img2.image(sc, caption="Spectral Centroid (Brightness)")
