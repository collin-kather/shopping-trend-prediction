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
ITEMS_PER_TRIAL = 20
TIME_LIMIT = 10
IMAGE_FOLDER = "images"
OUTPUT_DIR = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Feature Extraction ----------
def extract_index(filename):
    try:
        return int(os.path.splitext(filename)[0])  # "23.jpg" → 23
    except:
        return -1

# ---------- Session State ----------
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.user_id = str(uuid.uuid4())[:8]
    st.session_state.start_time = time.time()
    st.session_state.refresh_images = True
    st.session_state.model_trained = False

# ---------- Get Image Set ----------
def get_all_images():
    all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]
    return sorted(all_images, key=extract_index)

def get_trial_images():
    all_images = get_all_images()
    if len(all_images) < ITEMS_PER_TRIAL:
        st.error(f"❌ Not enough images. You need at least {ITEMS_PER_TRIAL}, but found {len(all_images)}.")
        st.stop()
    return random.sample(all_images, ITEMS_PER_TRIAL)

# ---------- ML Training ----------
def train_model(data):
    records = []
    for trial in data:
        for img in trial["options"]:
            features = {"index": extract_index(img)}
            features["chosen"] = 1 if img == trial["selection"] else 0
            records.append(features)
    df = pd.DataFrame(records)
    if df["chosen"].sum() == 0:
        return None
    X = df[["index"]]
    y = df["chosen"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# ---------- Prediction ----------
def predict_choice(model, image_list):
    df = pd.DataFrame({"index": [extract_index(f) for f in image_list]})
    preds = model.predict_proba(df)[:, 1]
    best_idx = preds.argmax()
    return image_list[best_idx], preds[best_idx]

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("🧠 Choose One (AI-Powered)")

if st.session_state.trial_index < NUM_TRIALS:
    trial_num = st.session_state.trial_index + 1
    remaining_time = TIME_LIMIT - int(time.time() - st.session_state.start_time)
    st.markdown(f"### Trial {trial_num} of {NUM_TRIALS}")
    st.markdown(f"⏳ Time left: **{remaining_time} seconds**")

    # Load current image set
    if st.session_state.refresh_images:
        st.session_state.current_images = get_trial_images()
        st.session_state.refresh_images = False
        st.session_state.start_time = time.time()

    # ML prediction (after 30 trials)
    predicted_img = None
    if trial_num > TRAIN_AFTER and not st.session_state.model_trained:
        st.session_state.model = train_model(st.session_state.choices)
        st.session_state.model_trained = True

    if trial_num > TRAIN_AFTER and st.session_state.model:
        predicted_img, pred_score = predict_choice(st.session_state.model, st.session_state.current_images)
        st.info(f"🤖 Predicted: `{predicted_img}` (Confidence: {pred_score:.2f})")

    # Render images
    cols = st.columns(5)
    selected = None
    for i, img in enumerate(st.session_state.current_images):
        with cols[i % 5]:
            st.image(f"{IMAGE_FOLDER}/{img}", use_container_width=True)
            if st.button("Choose", key=f"choose_{i}"):
                selected = img

    # Handle timeout or user choice
    if remaining_time <= 0 or selected:
        if remaining_time <= 0:
            selection = None
        else:
            selection = selected

        response_time = round(time.time() - st.session_state.start_time, 2)

        st.session_state.choices.append({
            "trial": trial_num,
            "options": st.session_state.current_images,
            "selection": selection,
            "predicted": predicted_img,
            "response_time": response_time,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": st.session_state.user_id
        })

        st.session_state.trial_index += 1
        st.session_state.refresh_images = True
        st.experimental_rerun()

else:
    st.success("✅ All done! Here's your result:")
    df = pd.DataFrame(st.session_state.choices)
    df["correct_prediction"] = df["selection"] == df["predicted"]
    filename = f"{OUTPUT_DIR}/results_{st.session_state.user_id}.csv"
    df.to_csv(filename, index=False)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), file_name="your_choices.csv", mime="text/csv")
