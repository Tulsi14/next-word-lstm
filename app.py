import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NextWord · LSTM Predictor",
    page_icon="🔮",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem; max-width: 720px; }

/* ── Hero ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: #7c6fff;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(2.2rem, 6vw, 3.4rem);
    font-weight: 700;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a89eff 60%, #6c5ce7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.75rem;
}
.hero-sub {
    font-size: 1rem;
    color: #7a7890;
    font-weight: 400;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #2e2b45, transparent);
    margin: 2rem 0;
}

/* ── Input card ── */
.card {
    background: #13111e;
    border: 1px solid #2a2640;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #5c5880;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* Override Streamlit text input */
.stTextInput > div > div > input {
    background: #0d0b18 !important;
    border: 1.5px solid #2e2b45 !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1rem !important;
    caret-color: #7c6fff;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #7c6fff !important;
    box-shadow: 0 0 0 3px rgba(124, 111, 255, 0.12) !important;
}
.stTextInput label { display: none; }

/* Override Streamlit button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6c5ce7, #a89eff) !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s;
    cursor: pointer;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0px);
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #13111e 0%, #1a1630 100%);
    border: 1px solid #3d2fa0;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-top: 1.25rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(to right, #6c5ce7, #a89eff, #6c5ce7);
}
.result-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #7c6fff;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.result-sentence {
    font-size: 1.35rem;
    font-weight: 500;
    line-height: 1.5;
    color: #c9c5e8;
    margin-bottom: 0.4rem;
}
.result-word {
    color: #a89eff;
    font-weight: 700;
    font-size: 1.45rem;
    background: rgba(108, 92, 231, 0.15);
    border-radius: 6px;
    padding: 0 6px;
}
.result-hint {
    font-size: 0.8rem;
    color: #4a4766;
    margin-top: 1rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── How it works ── */
.how-wrap {
    margin-top: 2.5rem;
}
.how-title {
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    color: #5c5880;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
    text-align: center;
}
.steps {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    flex-wrap: wrap;
}
.step {
    background: #13111e;
    border: 1px solid #2a2640;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    flex: 1;
    min-width: 140px;
    max-width: 200px;
    text-align: center;
}
.step-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}
.step-text {
    font-size: 0.82rem;
    color: #7a7890;
    line-height: 1.45;
}

/* ── Error state ── */
.error-card {
    background: #160d0d;
    border: 1px solid #3d1515;
    border-radius: 16px;
    padding: 1.25rem 1.75rem;
    margin-top: 1rem;
    color: #e07070;
    font-size: 0.9rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.75rem;
    color: #3a3755;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & tokenizer ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    model = load_model('next_word_lstm.h5')
    working_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(working_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">LSTM · Early Stopping · Shakespeare Corpus</div>
    <h1 class="hero-title">NextWord</h1>
    <p class="hero-sub">Type a phrase and the model predicts what comes next — powered by a character-aware LSTM trained on literary text.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-label">Your phrase</div>', unsafe_allow_html=True)
input_text = st.text_input(
    label="phrase",
    value="To be or not to",
    placeholder="e.g. To be or not to",
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("🔮  Predict next word", use_container_width=True)


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not input_text.strip():
        st.markdown('<div class="error-card">⚠ Please enter some text before predicting.</div>', unsafe_allow_html=True)
    else:
        with st.spinner(""):
            try:
                model, tokenizer = load_resources()
                max_seq_len = model.input_shape[1] + 1
                next_word = predict_next_word(model, tokenizer, input_text.strip(), max_seq_len)

                if next_word:
                    full_sentence = input_text.strip() + " " + next_word
                    display_sentence = input_text.strip() + " <span class='result-word'>" + next_word + "</span>"

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-eyebrow">Predicted continuation</div>
                        <div class="result-sentence">{display_sentence}</div>
                        <div class="result-hint">model confidence → argmax over {len(tokenizer.word_index):,} tokens</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-card">⚠ Could not predict a next word. Try a different phrase.</div>', unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="error-card">⚠ Error loading model: <code>{e}</code></div>', unsafe_allow_html=True)


# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-wrap">
    <div class="how-title">How it works</div>
    <div class="steps">
        <div class="step">
            <div class="step-icon">✍️</div>
            <div class="step-text">You type a sequence of words</div>
        </div>
        <div class="step">
            <div class="step-icon">🔢</div>
            <div class="step-text">Tokenizer maps words to indices</div>
        </div>
        <div class="step">
            <div class="step-icon">🧠</div>
            <div class="step-text">LSTM predicts the most likely next token</div>
        </div>
        <div class="step">
            <div class="step-icon">💬</div>
            <div class="step-text">Top token is decoded and shown to you</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">built with tensorflow · keras · streamlit</div>', unsafe_allow_html=True)
