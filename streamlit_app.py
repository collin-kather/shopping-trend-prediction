import streamlit as st
import random
import time
import pandas as pd
from datetime import datetime
import uuid
import os
from sklearn.ensemble import RandomForestClassifier

# ---------- CONFIG ----------
NUM_TRIALS = 40
TRAIN_AFTER = 30
ITEMS_PER_TRIAL = 5
IMAGE_FOLDER = "images_cleaned"
FEATURE_FILE = "simple_image_features.csv"
OUTPUT_DIR = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- LOAD FEATURES ----------
features_df = pd.read_csv(FEATURE_FILE)
features_df["filename"] = features_df["filename"].astype(str)

# ---------- SESSION ----------
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.user_id = str(uuid.uuid4())[:8]
    st.session_state.refresh_images = True
    st.session_state.model_trained = False
    st.session_state.last_result = None

# ---------- IMAGE UTILS ----------
def get_all_images():
    return sorted(features_df["filename"].tolist())

def get_trial_images():
    all_images = get_all_images()
    if len(all_images) < ITEMS_PER_TRIAL:
        st.error(f"❌ Not enough images! Found {len(all_images)} but need at least {ITEMS_PER_TRIAL}.")
        st.stop()
    return random.sample(all_images, ITEMS_PER_TRIAL)

# ---------- ML UTILS ----------
def train_model(data):
    records = []
    for trial in data:
        for img in trial["options"]:
            f = features_df[features_df["filename"] == img]
            if not f.empty:
                row = f.iloc[0].to_dict()
                row["chosen"] = int(img == trial["selection"])
                records.append(row)
    df = pd.DataFrame(records)
    if df["chosen"].sum() == 0:
        return None
    X = df.drop(columns=["filename", "chosen"])
    y = df["chosen"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def predict_choice(model, image_list):
    rows = []
    for img in image_list:
        row = features_df[features_df["filename"] == img]
        if not row.empty:
            rows.append(row.drop(columns=["filename"]).values[0])
    if not rows:
        return None, None
    X_pred = pd.DataFrame(rows)
    preds = model.predict_proba(X_pred)[:, 1]
    best_idx = preds.argmax()
    return image_list[best_idx], preds[best_idx]

# ---------- STYLING ----------
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    body {
        background-color: black;
        color: #00FF00;
    }
    .retail-card {
        border: 1px solid #00FF00;
        border-radius: 8px;
        padding: 6px;
        text-align: center;
        background-color: black;
    }
    .stButton>button {
        background-color: #111;
        color: #0f0;
        border: 1px solid #0f0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("🛒 Retro Grocery Shelf Game")

# ---------- DISPLAY LAST RESULT ----------
if st.session_state.last_result and st.session_state.model_trained:
    chosen, predicted, confidence = st.session_state.last_result

    if chosen and predicted:
        st.markdown("## Results (AI vs. Human)")
        cols = st.columns(2)

        with cols[0]:
            st.markdown("🛍️ **You chose:**")
            st.image(f"{IMAGE_FOLDER}/{chosen}", width=200)

        with cols[1]:
            st.markdown(f"🤖 **AI predicted:** (Confidence: `{confidence:.2f}`)")
            try:
                st.image(f"{IMAGE_FOLDER}/{predicted}", width=200)
            except:
                st.warning("⚠️ Could not load predicted image.")
    else:
        st.info("No prediction made yet. Keep going!")

# ---------- MAIN TRIAL ----------
if st.session_state.trial_index < NUM_TRIALS:
    trial_num = st.session_state.trial_index + 1
    st.markdown(f"### Trial {trial_num} of {NUM_TRIALS}")

    # Get images
    if st.session_state.refresh_images:
        st.session_state.current_images = get_trial_images()
        st.session_state.refresh_images = False

    # ML model
    predicted_img, pred_score = None, None
    if trial_num > TRAIN_AFTER:
        if not st.session_state.model_trained:
            st.session_state.model = train_model(st.session_state.choices)
            st.session_state.model_trained = True
        if st.session_state.model:
            predicted_img, pred_score = predict_choice(st.session_state.model, st.session_state.current_images)

    # Display images in one row
    cols = st.columns(ITEMS_PER_TRIAL)
    selected = None
    for i, img in enumerate(st.session_state.current_images):
        with cols[i]:
            st.markdown(f"<div class='retail-card'>", unsafe_allow_html=True)
            st.image(f"{IMAGE_FOLDER}/{img}", use_container_width=True)
            if st.button("Choose", key=f"choose_{i}"):
                selected = img
            st.markdown("</div>", unsafe_allow_html=True)

    # Handle selection
    if selected:
        st.session_state.choices.append({
            "trial": trial_num,
            "options": st.session_state.current_images,
            "selection": selected,
            "predicted": predicted_img,
            "response_time": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": st.session_state.user_id
        })

        st.session_state.last_result = (selected, predicted_img, pred_score)
        st.session_state.trial_index += 1
        st.session_state.refresh_images = True
        st.rerun()

# ---------- END ----------
else:
    st.success("✅ You're done! Here are your results:")
    df = pd.DataFrame(st.session_state.choices)
    df["correct_prediction"] = df["selection"] == df["predicted"]
    filename = f"{OUTPUT_DIR}/results_{st.session_state.user_id}.csv"
    df.to_csv(filename, index=False)
    st.dataframe(df)
    st.download_button("Download Results CSV", data=df.to_csv(index=False), file_name="your_choices.csv", mime="text/csv")
