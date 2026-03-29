"""
=============================================================
  Deepfake Audio Detection System — Streamlit Web App
  VISHING DEFENSE PROJECT
=============================================================
  Run with:  streamlit run streamlit_app.py
=============================================================
"""

# ── Standard Library ──────────────────────────────────────
import os
import io
import pickle
import tempfile
import warnings
warnings.filterwarnings("ignore")

# ── Data & Audio ──────────────────────────────────────────
import numpy as np
import librosa
import librosa.display

# ── Visualisation ─────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Web Framework ─────────────────────────────────────────
import streamlit as st


# ════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION  (must be the FIRST Streamlit call)
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Deepfake Voice Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)


# ════════════════════════════════════════════════════════════
#  CUSTOM CSS  ─ dark cybersecurity theme
# ════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Import Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c14;
    color: #c9d1d9;
}

/* ── Main container ── */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1100px;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0a1628 100%);
    border: 1px solid #1f3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;  left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 70% 40%, rgba(88,166,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0 0 0.4rem 0;
}
.hero-title span { color: #58a6ff; }
.hero-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #3fb950;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Upload zone ── */
.upload-zone {
    background: #0d1117;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.3s;
}

/* ── Result cards ── */
.result-card {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    border: 1px solid;
    text-align: center;
    animation: fadeUp 0.5s ease-out;
}
.result-real {
    background: linear-gradient(135deg, #0a2318, #0d2818);
    border-color: #2ea043;
    box-shadow: 0 0 40px rgba(46,160,67,0.15);
}
.result-fake {
    background: linear-gradient(135deg, #2d0a0a, #3a0d0d);
    border-color: #da3633;
    box-shadow: 0 0 40px rgba(218,54,51,0.15);
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
}
.result-real  .result-label { color: #3fb950; }
.result-fake  .result-label { color: #f85149; }
.result-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 0.5rem;
}

/* ── Confidence bar ── */
.conf-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.conf-track {
    background: #21262d;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.conf-fill-real { background: linear-gradient(90deg, #238636, #3fb950); height: 100%; border-radius: 999px; }
.conf-fill-fake { background: linear-gradient(90deg, #b91c1c, #ef4444); height: 100%; border-radius: 999px; }

/* ── Info badges ── */
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin: 0 4px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-blue  { background: #1f3a5f; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-green { background: #0a2318; color: #3fb950; border: 1px solid #2ea043; }
.badge-red   { background: #2d0a0a; color: #f85149; border: 1px solid #6e3738; }

/* ── Metric box ── */
.metric-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-key {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li { color: #8b949e; font-size: 0.85rem; }
[data-testid="stSidebar"] h3 { color: #c9d1d9; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}
.scanning { animation: pulse 1.2s ease-in-out infinite; color: #58a6ff; }

/* ── Hide Streamlit default UI elements ── */
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
header[data-testid="stHeader"] { display: none !important; }
footer { visibility: hidden !important; }
footer::after { content: none !important; }
#MainMenu { visibility: hidden !important; display: none !important; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2.5rem;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff);
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(56,139,253,0.35);
}
.stFileUploader { border-radius: 12px; }
div[data-testid="stAudio"] { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  UTILITY — LOAD MODEL
# ════════════════════════════════════════════════════════════

@st.cache_resource   # cache so model is loaded once, not on every re-run
def load_model(model_path="voice_model.pkl"):
    """
    Load the trained Random Forest model and scaler from disk.
    Returns (model, scaler) or (None, None) if file not found.
    """
    if not os.path.exists(model_path):
        return None, None
    try:
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        return payload["model"], payload["scaler"]
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


# ════════════════════════════════════════════════════════════
#  UTILITY — FEATURE EXTRACTION  (same as train_model.py)
# ════════════════════════════════════════════════════════════

def extract_features(file_path, n_mfcc=40):
    """Extract the same feature vector used during training."""
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050, duration=5.0)

        mfccs    = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        chroma   = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        zcr      = librosa.feature.zero_crossing_rate(y=audio)
        rms      = librosa.feature.rms(y=audio)

        def summarise(feat):
            return np.concatenate([
                np.mean(feat, axis=1),
                np.std(feat, axis=1),
                np.max(feat, axis=1)
            ])

        return np.concatenate([
            summarise(mfccs),
            summarise(chroma),
            summarise(contrast),
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)]
        ]), audio, sample_rate

    except Exception as e:
        return None, None, None


# ════════════════════════════════════════════════════════════
#  UTILITY — PREDICT
# ════════════════════════════════════════════════════════════

def predict_audio(file_path, model, scaler):
    """
    Run the deepfake detector on a single audio file.

    Returns:
        label       (str)   : "Real Voice" or "Deepfake Voice"
        confidence  (float) : 0.0 – 1.0
        audio       (np.array)
        sample_rate (int)
    """
    features, audio, sample_rate = extract_features(file_path)

    if features is None:
        return None, None, None, None

    # Scale features using the same scaler fitted during training
    features_scaled = scaler.transform([features])

    # Get class probabilities — shape: [1, 2]
    probabilities = model.predict_proba(features_scaled)[0]

    # Index 0 = real, index 1 = fake
    predicted_class = model.predict(features_scaled)[0]
    confidence = float(probabilities[predicted_class])

    label = "Deepfake Voice" if predicted_class == 1 else "Real Voice"
    return label, confidence, audio, sample_rate


# ════════════════════════════════════════════════════════════
#  UTILITY — WAVEFORM PLOT
# ════════════════════════════════════════════════════════════

def plot_waveform(audio, sample_rate, is_fake):
    """
    Return a matplotlib figure of the audio waveform.
    Colour is green for real, red for fake.
    """
    color = "#f85149" if is_fake else "#3fb950"
    fill  = "#2d0a0a"  if is_fake else "#0a2318"

    fig, ax = plt.subplots(figsize=(9, 2.2))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    times = np.linspace(0, len(audio) / sample_rate, len(audio))
    ax.plot(times, audio, color=color, linewidth=0.6, alpha=0.9)
    ax.fill_between(times, audio, alpha=0.18, color=color)

    ax.set_xlabel("Time (seconds)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Amplitude",      color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#21262d")

    ax.set_title("Waveform Visualization", color="#c9d1d9", fontsize=10, pad=8)
    plt.tight_layout()
    return fig


def plot_mfcc(file_path, is_fake):
    """
    Return a matplotlib figure of the MFCC heatmap.
    Colour map is red-tinted for fake, green-tinted for real.
    """
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=5.0)
    except Exception:
        return None

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    cmap  = "RdPu" if is_fake else "GnBu"

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    img = librosa.display.specshow(
        mfccs, sr=sr, x_axis="time", ax=ax, cmap=cmap
    )
    fig.colorbar(img, ax=ax, format="%+.1f", label="Coefficient value")
    ax.set_ylabel("MFCC Coefficient", color="#8b949e", fontsize=9)
    ax.set_xlabel("Time (seconds)",   color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#21262d")
    ax.set_title("MFCC — Mel-Frequency Cepstral Coefficients",
                 color="#c9d1d9", fontsize=10, pad=8)
    plt.tight_layout()
    return fig



# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🛡️ System Status")
    model, scaler = load_model()

    if model is not None:
        st.markdown("""
        <span class="badge badge-green">● MODEL LOADED</span>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <span class="badge badge-red">✗ NO MODEL</span>
        """, unsafe_allow_html=True)
        st.markdown("Run `python train_model.py` first.")

    st.markdown("---")
    st.markdown("### 📖 How It Works")
    st.markdown("""
- Audio is loaded and analysed
- Audio features (chroma, spectral contrast, ZCR, RMS) are extracted as a numerical fingerprint of the voice
- The **Random Forest** model compares these against patterns learned from thousands of real and fake audio samples
- A prediction + confidence score is returned
    """)

    st.markdown("---")
    st.markdown("### ⚠️ What is Vishing?")
    st.markdown("""
**Voice phishing (vishing)** is a social engineering attack where criminals use phone calls — increasingly with **AI-cloned voices** — to trick victims into revealing sensitive information.

This system helps identify whether a voice recording was generated by an AI.
    """)

    st.markdown("---")
    st.markdown("### 📁 Accepted Formats")
    st.markdown("""
- WAV &nbsp; *(best quality)*
- MP3
- FLAC
- OGG
    """)


# ════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════════════════════════

# ── Hero Header ───────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-subtitle">🔒 &nbsp; Cybersecurity AI Tool &nbsp;·&nbsp; Vishing Defense</div>
    <h1 class="hero-title">Deepfake <span>Voice</span> Detector</h1>
    <p style="color:#8b949e; margin:0; font-size:0.95rem;">
        Upload an audio recording to determine whether it is a <strong style="color:#3fb950">real human voice</strong>
        or an <strong style="color:#f85149">AI-generated deepfake</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Model-not-found warning ────────────────────────────────
if model is None:
    st.warning(
        "⚠️  No trained model found (`voice_model.pkl`).  \n"
        "Run **`python train_model.py`** in your terminal first, "
        "then refresh this page.",
        icon="⚠️"
    )

# ── Upload Section ────────────────────────────────────────
st.markdown("#### 📂 Upload Audio File")
uploaded_file = st.file_uploader(
    label="Drop a .wav / .mp3 / .flac / .ogg file here",
    type=["wav", "mp3", "flac", "ogg"],
    help="The file will be analysed locally — nothing is sent to an external server.",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # ── Audio Player ──────────────────────────────────────
    st.markdown("#### 🎧 Audio Preview")
    st.audio(uploaded_file, format="audio/wav")

    # ── File metadata ─────────────────────────────────────
    file_size_kb = round(uploaded_file.size / 1024, 1)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{uploaded_file.name.split('.')[-1].upper()}</div>
            <div class="metric-key">Format</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{file_size_kb} KB</div>
            <div class="metric-key">File Size</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{uploaded_file.name[:18]}{'…' if len(uploaded_file.name)>18 else ''}</div>
            <div class="metric-key">Filename</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Detect Button ─────────────────────────────────────
    detect_btn = st.button("🔍  Detect Voice", disabled=(model is None))

    if detect_btn:
        if model is None:
            st.error("Please train the model first: `python train_model.py`")
        else:
            # Save to temp file so librosa can read it
            suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # ── Scanning animation ──────────────────────
            with st.spinner(""):
                st.markdown(
                    '<p class="scanning">⬡ Analysing audio features …</p>',
                    unsafe_allow_html=True
                )
                label, confidence, audio, sr = predict_audio(tmp_path, model, scaler)

            # ── Show result ─────────────────────────────
            if label is None:
                st.error("❌  Could not process the audio file. "
                         "Ensure it is a valid audio recording.")
            else:
                is_fake = (label == "Deepfake Voice")

                # Result card
                card_class = "result-fake" if is_fake else "result-real"
                icon       = "🤖" if is_fake else "✅"
                conf_pct   = round(confidence * 100, 1)
                fill_class = "conf-fill-fake" if is_fake else "conf-fill-real"

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="result-icon">{icon}</div>
                    <p class="result-label">{label}</p>
                    <p class="result-desc">
                        {'AI-generated audio detected — potential vishing threat' if is_fake
                         else 'Natural human vocal patterns detected'}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                st.markdown(f"""
                <br>
                <div class="conf-label">Confidence Score</div>
                <div class="conf-track">
                    <div class="{fill_class}" style="width:{conf_pct}%"></div>
                </div>
                <p style="font-family:'JetBrains Mono',monospace; font-size:0.9rem;
                           color:{'#f85149' if is_fake else '#3fb950'}; margin-top:4px;">
                    {conf_pct}%
                </p>
                """, unsafe_allow_html=True)

                # Threat level banner
                if is_fake and confidence >= 0.85:
                    st.error("🚨  HIGH CONFIDENCE DEEPFAKE — Do not trust this voice.")
                elif is_fake and confidence >= 0.65:
                    st.warning("⚠️  POSSIBLE DEEPFAKE — Treat this call with caution.")
                elif not is_fake and confidence >= 0.80:
                    st.success("✅  HIGH CONFIDENCE REAL VOICE — Natural speech patterns confirmed.")
                else:
                    st.info("ℹ️  LOW CONFIDENCE — Result is uncertain. Consider further analysis.")

                # ── Visualisations ───────────────────────
                st.markdown("---")
                st.markdown("#### 📊 Audio Analysis")

                st.markdown("**🌊 Waveform**")
                st.markdown(
                    "The waveform shows amplitude over time. "
                    "Deepfakes often have subtly unnatural amplitude patterns."
                )
                fig_wave = plot_waveform(audio, sr, is_fake)
                st.pyplot(fig_wave, use_container_width=True)
                plt.close(fig_wave)

                st.markdown("**🎛️ MFCC Heatmap**")
                st.markdown(
                    "MFCCs capture the voice's spectral fingerprint over time. "
                    "AI-generated voices often show unnaturally uniform or repetitive patterns across coefficients."
                )
                fig_mfcc = plot_mfcc(tmp_path, is_fake)
                if fig_mfcc:
                    st.pyplot(fig_mfcc, use_container_width=True)
                    plt.close(fig_mfcc)

                # ── Technical details expander ────────────
                with st.expander("🔧 Technical Details"):
                    st.markdown(f"""
| Property | Value |
|---|---|
| Prediction | `{label}` |
| Confidence | `{conf_pct}%` |
| Sample Rate | `{sr} Hz` |
| Audio Length | `{len(audio)/sr:.2f} seconds` |
| Feature Dimensions | `154` |
| Model | `RandomForestClassifier (200 trees)` |
| Feature Types | `Chroma, Spectral Contrast, ZCR, RMS` |
                    """)

            # Clean up temp file
            os.unlink(tmp_path)

else:
    # ── Empty state placeholder ───────────────────────────
    st.markdown("""
    <div class="upload-zone">
        <p style="font-size:2.5rem; margin:0">🎙️</p>
        <p style="color:#8b949e; margin:0.5rem 0 0 0">
            Upload an audio file above to begin analysis
        </p>
        <p style="color:#3d444d; font-family:'JetBrains Mono',monospace;
                  font-size:0.75rem; margin-top:0.3rem">
            .wav · .mp3 · .flac · .ogg
        </p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#3d444d; font-family:'JetBrains Mono',monospace;
          font-size:0.72rem;">
    🛡️ Deepfake Audio Detection System &nbsp;·&nbsp; Vishing Defense Project &nbsp;·&nbsp;
    Built with Python · librosa · scikit-learn · Streamlit
</p>
""", unsafe_allow_html=True)
