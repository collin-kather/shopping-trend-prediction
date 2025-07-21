import streamlit as st
import pandas as pd
import os
import random
import time
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import uuid

# ---------- CONFIG ----------
IMAGE_FOLDER = "images"
CSV_FEATURES = "simple_image_features.csv"
NUM_TRIALS = 40
TRAIN_AFTER = 30
ITEMS_PER_TRIAL = 5

# ---------- LOAD FEATURES ----------
@st.cache_data
def load_features():
    df = pd.read_csv(CSV_FEATURES)
    df["index"] = df["image"].apply(lambda x: int(os.path.splitext(x)[0]))
    return df

feature_df = load_features()

# ---------- INIT SESSION ----------
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.refresh_images = True
    st.session_state.last_result = None
    st.session_state.user_id = str(uuid.uuid4())[:8]

# ---------- UTILS ----------
def get_all_images():
    return sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")])

def get_trial_images():
    all_images = get_all_images()
    return random.sample(all_images, ITEMS_PER_TRIAL)

def train_model(data):
    records = []
    for trial in data:
        for opt in trial["options"]:
            feat = feature_df[feature_df["image"] == opt]
            if feat.empty:
                continue
            row = feat.iloc[0].to_dict()
            row["chosen"] = int(opt == trial["selection"])
            records.append(row)
    df = pd.DataFrame(records)
    if df["chosen"].sum() == 0:
        return None
    X = df.drop(columns=["image", "chosen", "index"])
    y = df["chosen"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_choice(model, options):
    df_opts = feature_df[feature_df["image"].isin(options)]
    if df_opts.empty:
        return None, 0.0
    X = df_opts.drop(columns=["image", "index"])
    preds = model.predict_proba(X)[:, 1]
    best_idx = preds.argmax()
    return df_opts.iloc[best_idx]["image"], preds[best_idx]

# ---------- STYLING ----------
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .title-text { color: #00ff00; font-family: monospace; font-size: 30px; }
    .box-style {
        border: 2px solid #00ff00;
        border-radius: 12px;
        padding: 10px;
        background-color: #000000;
        color: #00ff00;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-text'>💾 Grocery Shelf Smart Selector</div>", unsafe_allow_html=True)
st.write("")

# ---------- LAST RESULT ----------
if st.session_state.last_result and st.session_state.trial_index >= TRAIN_AFTER:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🛍️ You Chose")
        st.image(f"{IMAGE_FOLDER}/{st.session_state.last_result['chosen']}", width=200)
    with col2:
        st.markdown("#### 🤖 We Predicted")
        predicted_path = f"{IMAGE_FOLDER}/{st.session_state.last_result['predicted']}"
        if st.session_state.last_result["predicted"] and os.path.exists(predicted_path):
            st.image(predicted_path, width=200)
            st.markdown(f"**Confidence:** `{st.session_state.last_result['confidence']:.2f}`")
            st.markdown("**Rationale:**")
            st.markdown("- Based on color, cartoon, or branding pattern")
            st.markdown("- Trained from your prior preferences")
        else:
            st.warning("Prediction not available or image not found.")

# ---------- MAIN TRIAL ----------
if st.session_state.trial_index < NUM_TRIALS:
    trial_num = st.session_state.trial_index + 1
    st.markdown(f"### Trial {trial_num} of {NUM_TRIALS}")

    if st.session_state.refresh_images:
        st.session_state.current_images = get_trial_images()
        st.session_state.refresh_images = False

    predicted_img, pred_score = None, None
    if trial_num > TRAIN_AFTER:
        if not st.session_state.model_trained:
            st.session_state.model = train_model(st.session_state.choices)
            st.session_state.model_trained = True
        if st.session_state.model:
            predicted_img, pred_score = predict_choice(st.session_state.model, st.session_state.current_images)

    cols = st.columns(ITEMS_PER_TRIAL)
    selected = None
    for i, img in enumerate(st.session_state.current_images):
        with cols[i]:
            st.image(f"{IMAGE_FOLDER}/{img}", use_container_width=True)
            if st.button("Choose", key=f"choose_{i}"):
                selected = img

    if selected:
        st.session_state.choices.append({
            "trial": trial_num,
            "options": st.session_state.current_images,
            "selection": selected
        })
        st.session_state.last_result = {
            "chosen": selected,
            "predicted": predicted_img,
            "confidence": pred_score if pred_score else 0.0
        }
        st.session_state.trial_index += 1
        st.session_state.refresh_images = True
        st.rerun()
else:
    st.success("🎉 You're done! Thanks for playing.")
    df = pd.DataFrame(st.session_state.choices)
    st.dataframe(df)
