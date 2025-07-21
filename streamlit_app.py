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
IMAGE_FOLDER = "images"
OUTPUT_DIR = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- SESSION SETUP ----------
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.user_id = str(uuid.uuid4())[:8]
    st.session_state.refresh_images = True
    st.session_state.model_trained = False
    st.session_state.last_result = None

# ---------- HELPER FUNCTIONS ----------
def extract_index(filename):
    try:
        return int(os.path.splitext(filename)[0])
    except:
        return -1

def get_all_images():
    return sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")], key=extract_index)

def get_trial_images():
    all_images = get_all_images()
    if len(all_images) < ITEMS_PER_TRIAL:
        st.error(f"❌ Not enough images! Found {len(all_images)} but need at least {ITEMS_PER_TRIAL}.")
        st.stop()
    return random.sample(all_images, ITEMS_PER_TRIAL)

def train_model(data):
    records = []
    for trial in data:
        for img in trial["options"]:
            records.append({
                "index": extract_index(img),
                "chosen": int(img == trial["selection"])
            })
    df = pd.DataFrame(records)
    if df["chosen"].sum() == 0:
        return None
    model = RandomForestClassifier(n_estimators=100)
    model.fit(df[["index"]], df["chosen"])
    return model

def predict_choice(model, image_list):
    df = pd.DataFrame({"index": [extract_index(f) for f in image_list]})
    preds = model.predict_proba(df)[:, 1]
    best_idx = preds.argmax()
    return image_list[best_idx], preds[best_idx]

# ---------- STYLING ----------
st.set_page_config(layout="wide", page_title="🛒 Grocery AI")
st.markdown("""
    <style>
    body {
        background-color: #000000;
        color: #00FF00;
        font-family: "Courier New", monospace;
    }
    .main {
        background-color: #000000;
    }
    .retail-card {
        border: 2px dashed #8000ff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 8px;
        background-color: #111111;
    }
    .choose-btn button {
        background-color: #00FF00 !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 8px !important;
    }
    h1, h2, h3, h4 {
        color: #00FF00;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>💾 90s Grocery-Shelf AI Simulator</h1>", unsafe_allow_html=True)

# ---------- SHOW LAST RESULT ----------
if st.session_state.last_result and st.session_state.trial_index > TRAIN_AFTER:
    chosen, predicted, confidence = st.session_state.last_result
    st.markdown("### 🔍 Prediction Recap")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🛍️ You chose: `{chosen or 'None'}`")
        st.image(f"{IMAGE_FOLDER}/{chosen}", width=250)
    with col2:
        st.info(f"🤖 AI predicted: `{predicted}` (Confidence: `{confidence:.2f}`)")
        st.image(f"{IMAGE_FOLDER}/{predicted}", width=250)

# ---------- MAIN TRIAL ----------
if st.session_state.trial_index < NUM_TRIALS:
    trial_num = st.session_state.trial_index + 1
    st.markdown(f"### 🧪 Trial {trial_num} of {NUM_TRIALS}")

    if st.session_state.refresh_images:
        st.session_state.current_images = get_trial_images()
        st.session_state.refresh_images = False

    # ML prediction
    predicted_img, pred_score = None, None
    if trial_num > TRAIN_AFTER:
        if not st.session_state.model_trained:
            st.session_state.model = train_model(st.session_state.choices)
            st.session_state.model_trained = True
        if st.session_state.model:
            predicted_img, pred_score = predict_choice(st.session_state.model, st.session_state.current_images)

    # Display images
    cols = st.columns(ITEMS_PER_TRIAL)
    selected = None
    for i, img in enumerate(st.session_state.current_images):
        with cols[i]:
            st.markdown(f"<div class='retail-card'>", unsafe_allow_html=True)
            st.image(f"{IMAGE_FOLDER}/{img}", use_container_width=True)
            if st.button("Choose", key=f"choose_{i}"):
                selected = img
            st.markdown("</div>", unsafe_allow_html=True)

    # Handle result
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
        st.session_state.model_trained = False
        st.rerun()

# ---------- END ----------
else:
    st.balloons()
    st.success("✅ All trials complete!")
    df = pd.DataFrame(st.session_state.choices)
    df["correct_prediction"] = df["selection"] == df["predicted"]
    filename = f"{OUTPUT_DIR}/results_{st.session_state.user_id}.csv"
    df.to_csv(filename, index=False)
    st.dataframe(df)
    st.download_button("⬇️ Download Results CSV", data=df.to_csv(index=False), file_name="your_choices.csv", mime="text/csv")
