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

# ---------- Feature Extraction from Filename ----------
def parse_features(filename):
    name = os.path.basename(filename).replace(".png", "")
    try:
        color, price, label, idx = name.split("_")
        return {
            "color": color,
            "price": int(price),
            "label": label,
            "index": int(idx)
        }
    except:
        return {"color": "unknown", "price": 0, "label": "none", "index": -1}

# ---------- Session State ----------
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.user_id = str(uuid.uuid4())[:8]
    st.session_state.start_time = time.time()
    st.session_state.refresh_images = True
    st.session_state.model_trained = False

# ---------- Trial Setup ----------
def get_all_images():
    return [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png")]

def get_trial_images():
    all_images = get_all_images()
    return random.sample(all_images, ITEMS_PER_TRIAL)

# ---------- ML Training ----------
def train_model(data):
    records = []
    for trial in data:
        for img in trial["options"]:
            features = parse_features(img)
            features["chosen"] = 1 if img == trial["selection"] else 0
            records.append(features)

    df = pd.DataFrame(records)
    if df["chosen"].sum() == 0:
        return None, None

    X = df.drop(columns=["chosen"])
    X = pd.get_dummies(X)
    y = df["chosen"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model, X.columns

# ---------- Prediction ----------
def predict_choice(model, columns, image_list):
    records = []
    for img in image_list:
        feats = parse_features(img)
        records.append(feats)
    df = pd.DataFrame(records)
    X = pd.get_dummies(df)

    for col in columns:
        if col not in X:
            X[col] = 0
    X = X[columns]

    preds = model.predict_proba(X)[:, 1]
    max_idx = preds.argmax()
    return image_list[max_idx], preds[max_idx]

# ---------- UI ----------
st.set_page_config(layout="wide")
st.title("🧠 Choose One (With AI Prediction)")

if st.session_state.trial_index < NUM_TRIALS:
    trial_num = st.session_state.trial_index + 1
    remaining_time = TIME_LIMIT - int(time.time() - st.session_state.start_time)
    st.markdown(f"### Trial {trial_num} of {NUM_TRIALS}")
    st.markdown(f"⏳ Time left: **{remaining_time} seconds**")

    # Load current images
    if st.session_state.refresh_images:
        st.session_state.current_images = get_trial_images()
        st.session_state.refresh_images = False
        st.session_state.start_time = time.time()

    # ML prediction (after 30 trials)
    predicted_img = None
    if trial_num > TRAIN_AFTER and not st.session_state.model_trained:
        st.session_state.model, st.session_state.model_columns = train_model(st.session_state.choices)
        st.session_state.model_trained = True

    if trial_num > TRAIN_AFTER and st.session_state.model:
        predicted_img, pred_score = predict_choice(st.session_state.model, st.session_state.model_columns, st.session_state.current_images)
        st.info(f"🤖 Our model predicts you'll pick: `{predicted_img}` (confidence: {pred_score:.2f})")

    # Show images
    cols = st.columns(5)
    selected = None
    for i, img in enumerate(st.session_state.current_images):
        with cols[i % 5]:
            st.image(f"{IMAGE_FOLDER}/{img}", use_column_width=True)
            if st.button("Choose", key=f"choose_{i}"):
                selected = img

    if remaining_time <= 0 or selected:
        selection = selected if selected else None
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
    st.success("✅ You're done! Here's how you did:")
    df = pd.DataFrame(st.session_state.choices)
    df["correct_prediction"] = df["selection"] == df["predicted"]
    filename = f"{OUTPUT_DIR}/results_{st.session_state.user_id}.csv"
    df.to_csv(filename, index=False)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), file_name="your_choices.csv", mime="text/csv")
