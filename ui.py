import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Intent Studio · Banking77",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "banking77_model"


# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b0d12;
        --panel: #12151d;
        --line: rgba(255,255,255,.09);
        --text: #f3f5f8;
        --muted: #929aaa;
        --accent: #b9f36b;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(185,243,107,.075), transparent 32%),
            radial-gradient(ellipse at 100% 15%, rgba(111,132,255,.07), transparent 28%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11,13,18,.78);
    }

    [data-testid="stSidebar"] {
        background: #0e1016;
        border-right: 1px solid var(--line);
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.04em;
    }

    .hero-kicker {
        color: var(--accent);
        font-family: 'DM Mono', monospace;
        font-size: .76rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .65rem;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.5rem, 5vw, 4.4rem);
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -.065em;
        color: var(--text);
        margin: 0;
    }

    .hero-title span {
        color: var(--accent);
    }

    .hero-sub {
        max-width: 650px;
        color: var(--muted);
        font-size: 1.04rem;
        line-height: 1.7;
        margin-top: 1rem;
    }

    .section-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .12em;
        font-family: 'DM Mono', monospace;
        font-size: .72rem;
        margin-bottom: .7rem;
    }

    .surface {
        background: linear-gradient(145deg, rgba(23,27,37,.96), rgba(17,20,28,.96));
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.25rem 1.35rem;
    }

    .mini-stat-label {
        color: var(--muted);
        font-size: .82rem;
    }

    .mini-stat-value {
        color: var(--text);
        font-family: 'Manrope', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -.04em;
        margin-top: .15rem;
    }

    .muted {
        color: var(--muted);
    }

    .intent-pill {
        display: inline-block;
        background: rgba(185,243,107,.1);
        color: var(--accent);
        border: 1px solid rgba(185,243,107,.24);
        border-radius: 999px;
        padding: .32rem .65rem;
        font: .76rem 'DM Mono', monospace;
    }

    .result-intent {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: clamp(1.35rem, 3vw, 2.1rem);
        letter-spacing: -.04em;
        overflow-wrap: anywhere;
        color: var(--text);
    }

    .readme {
        color: #c1c7d2;
        font-size: .93rem;
        line-height: 1.75;
    }

    div.stButton > button {
        border-radius: 12px;
        border: 1px solid var(--accent);
        background: var(--accent);
        color: #10140b;
        font-weight: 800;
        min-height: 3rem;
        transition: transform .15s ease, filter .15s ease;
    }

    div.stButton > button:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
    }

    div[data-testid="stTextArea"] textarea {
        background: #0e1118;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 14px;
        color: var(--text);
        line-height: 1.6;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }

    div[data-testid="stMetric"] {
        background: #11151d;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--line);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load saved model artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = MODEL_DIR / "ann.keras"
    vectorizer_path = MODEL_DIR / "tfidf.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"

    missing = [
        path.name
        for path in (model_path, vectorizer_path, encoder_path)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {', '.join(missing)}. "
            "Place them inside the banking77_model folder."
        )

    model = keras.models.load_model(model_path)
    vectorizer = joblib.load(vectorizer_path)
    encoder = joblib.load(encoder_path)

    model_features = model.input_shape[-1]
    vectorizer_features = len(vectorizer.vocabulary_)

    if model_features != vectorizer_features:
        raise ValueError(
            f"Model expects {model_features:,} features, but the "
            f"vectorizer produces {vectorizer_features:,}. "
            "Load matching artifacts or retrain the model."
        )

    if model.output_shape[-1] != len(encoder.classes_):
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs, but the encoder "
            f"contains {len(encoder.classes_)} classes."
        )

    return model, vectorizer, encoder


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return text


# --------------------------------------------------
# Sidebar: README and intent directory
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ◈ Intent Studio")
    st.caption("BANKING77 · ANN + TF-IDF")
    st.divider()

    with st.expander("README · About this project", expanded=True):
        st.markdown(
            """
            <div class="readme">
            <b>What it does</b><br>
            Classifies a banking-related message into an intent, such as
            a card issue, cash withdrawal problem, or account question.

            <br><br><b>How it works</b><br>
            1. Lowercase the message and remove punctuation.<br>
            2. Convert text into TF-IDF features.<br>
            3. Pass features through a trained ANN.<br>
            4. Decode the highest-scoring output into an intent label.

            <br><br><b>Technology</b><br>
            Python · Streamlit · scikit-learn · TensorFlow/Keras

            <br><br><b>Important</b><br>
            Scores are model probabilities, not calibrated certainty.
            This is a learning/demo project, not a financial service.
            </div>
            """,
            unsafe_allow_html=True,
        )

    try:
        model, vectorizer, encoder = load_artifacts()
        intents = sorted(str(label) for label in encoder.classes_)

        st.divider()
        st.markdown("### Defined intents")
        st.caption(f"{len(intents)} classes loaded from the encoder")

        with st.container(height=280):
            for intent in intents:
                st.markdown(f"`{intent}`")

    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")
        st.stop()


# --------------------------------------------------
# Main page header
# --------------------------------------------------
st.markdown(
    '<div class="hero-kicker">TEXT CLASSIFICATION / 01</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">Understand the <span>intent.</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-sub">A neural network trained to recognize what '
    'a customer is trying to do or resolve in a banking conversation.</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Project statistics
# --------------------------------------------------
st.write("")

stat1, stat2, stat3 = st.columns(3)

with stat1:
    st.markdown(
        f"""
        <div class="surface">
            <div class="mini-stat-label">INTENT CLASSES</div>
            <div class="mini-stat-value">{len(encoder.classes_)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat2:
    st.markdown(
        f"""
        <div class="surface">
            <div class="mini-stat-label">TF-IDF FEATURES</div>
            <div class="mini-stat-value">{len(vectorizer.vocabulary_):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat3:
    st.markdown(
        """
        <div class="surface">
            <div class="mini-stat-label">PREDICTION MODE</div>
            <div class="mini-stat-value">ANN / Softmax</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Input and output panels
# --------------------------------------------------
st.write("")

left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.markdown(
        '<div class="section-label">01 / Input message</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        example = st.selectbox(
            "Try an example",
            [
                "Write my own message",
                "My card was charged twice.",
                "I forgot my PIN and need to reset it.",
                "I made a cash withdrawal but the money never came out.",
                "How long will my bank transfer take?",
                "I want to cancel a payment.",
            ],
        )

        examples = {
            "My card was charged twice.": "My card was charged twice.",
            "I forgot my PIN and need to reset it.": (
                "I forgot my PIN and need to reset it."
            ),
            "I made a cash withdrawal but the money never came out.": (
                "I made a cash withdrawal but the money never came out."
            ),
            "How long will my bank transfer take?": (
                "How long will my bank transfer take?"
            ),
            "I want to cancel a payment.": "I want to cancel a payment.",
        }

        initial_text = examples.get(example, "")

        user_text = st.text_area(
            "Your message",
            value=initial_text,
            height=155,
            placeholder=(
                "e.g. My card payment was declined even though "
                "I have enough money."
            ),
            label_visibility="collapsed",
            key="user_message",
        )

        st.caption(
            f"{len(user_text)} characters · Keep it to one customer "
            "request for a clearer prediction."
        )

        predict = st.button(
            "Classify message  →",
            use_container_width=True,
        )


with right:
    st.markdown(
        '<div class="section-label">02 / Model output</div>',
        unsafe_allow_html=True,
    )

    if predict:
        if not user_text.strip():
            st.warning("Enter a message before running the classifier.")

        else:
            cleaned_text = preprocess(user_text)

            features = vectorizer.transform([cleaned_text])
            features = features.astype(np.float32).toarray()

            with st.spinner("Classifying message…"):
                probabilities = model.predict(
                    features,
                    verbose=0,
                )[0]

            predicted_index = int(np.argmax(probabilities))

            predicted_intent = str(
                encoder.inverse_transform([predicted_index])[0]
            )

            confidence = float(probabilities[predicted_index])

            top_indices = np.argsort(probabilities)[-5:][::-1]

            st.markdown(
                f"""
                <div class="surface">
                    <div class="section-label">PREDICTED INTENT</div>
                    <div class="result-intent">
                        {predicted_intent.replace("_", " ")}
                    </div>
                    <div style="margin-top:.8rem">
                        <span class="intent-pill">{predicted_intent}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.metric("Top-class model score", f"{confidence:.1%}")
            st.progress(confidence)

            st.caption(
                "A high score does not guarantee that the prediction is correct."
            )

            with st.expander("Explore top 5 predictions", expanded=True):
                results = pd.DataFrame(
                    {
                        "Intent": encoder.inverse_transform(top_indices),
                        "Score": probabilities[top_indices],
                    }
                )

                results["Score"] = results["Score"].map(
                    lambda score: f"{score:.1%}"
                )

                st.dataframe(
                    results,
                    hide_index=True,
                    use_container_width=True,
                )

    else:
        st.markdown(
            """
            <div class="surface" style="
                min-height:260px;
                display:flex;
                flex-direction:column;
                justify-content:center;
            ">
                <div class="section-label">WAITING FOR INPUT</div>

                <div style="
                    font-family:Manrope,sans-serif;
                    font-size:1.55rem;
                    font-weight:800;
                    letter-spacing:-.04em;
                ">
                    Your result will appear here.
                </div>

                <p class="muted" style="line-height:1.7;margin-top:.6rem;">
                    Enter a message on the left and run the classifier to see
                    its predicted intent and the five highest-scoring classes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="muted" style="font-size:.8rem;">'
    'INTENT STUDIO · BANKING77 DEMO · '
    'Predictions are generated locally from saved model artifacts.'
    '</div>',
    unsafe_allow_html=True,
)