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
TOTAL_TIME_LIMIT = 120  # 2 minutes for first 30 rounds
IMAGE_FOLDER = "images"
OUTPUT_DIR = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- UTILS ----------
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
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: black !important;
        color: #00FF00 !important;
        font-family: 'Courier New', Courier, monospace;
    }
    .retail-card {
        border: 2px dashed #00FF00;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #111;
    }
    .neon-box {
        border: 1px solid #00FF00;
        padding: 10px;
        margin-top: 10px;
        background-color: #000;
        border-radius: 10px;
    }
    .timer {
        font-size: 24px;
        font-weight: bold;
        color: #00FF00;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- SESSION INIT ----------
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.user_id = str(uuid.uuid4())[:8]
    st.session_state.refresh_images = True
    st.session_state.model_trained = False
    st.session_state.last_result = None
    st.session_state.timer_started = False
    st.session_state.start_time = 0

# ---------- HEADER ----------
st.title("💾 90s Grocery AI Simulator")
st.markdown("Welcome to the **retro shelf selection challenge**. Choose one item per row. Let's see if the AI learns your taste...")

# ---------- START BUTTON ----------
if not st.session_state.timer_started:
    if st.button("💾 Begin Simulation"):
        st.session_state.timer_started = True
        st.session_state.start_time = time.time()
        st.rerun()
    st.stop()

# ---------- MAIN GAME ----------
elapsed = int(time.time() - st.session_state.start_time)
time_remaining = max(TOTAL_TIME_LIMIT - elapsed, 0)

if st.session_state.trial_index < NUM_TRIALS:
    trial_num = st.session_state.trial_index + 1
    st.markdown(f"### Trial {trial_num} of {NUM_TRIALS}")

    if trial_num <= TRAIN_AFTER:
        st.markdown(f"⏳ <div class='timer'>Time left for training phase: {time_remaining} seconds</div>", unsafe_allow_html=True)
        if time_remaining <= 0:
            st.warning("⏰ Time's up! But you can still finish all 30 rounds.")

    if st.session_state.refresh_images:
        st.session_state.current_images = get_trial_images()
        st.session_state.refresh_images = False

    # ML Prediction
    predicted_img, pred_score = None, None
    if trial_num > TRAIN_AFTER and not st.session_state.model_trained:
        st.session_state.model = train_model(st.session_state.choices)
        st.session_state.model_trained = True

    if trial_num > TRAIN_AFTER and st.session_state.model:
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

    if selected:
        st.session_state.choices.append({
            "trial": trial_num,
            "options": st.session_state.current_images,
            "selection": selected,
            "predicted": predicted_img,
            "response_time": elapsed,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": st.session_state.user_id
        })

        st.session_state.last_result = (selected, predicted_img, pred_score)
        st.session_state.trial_index += 1
        st.session_state.refresh_images = True
        st.rerun()

    # Show prediction ONLY after training phase
    if trial_num > TRAIN_AFTER and st.session_state.last_result:
        chosen, predicted, confidence = st.session_state.last_result
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### You Chose:")
            st.image(f"{IMAGE_FOLDER}/{chosen}", caption=chosen, use_container_width=True)
        with col2:
            st.markdown("### We Predicted:")
            if predicted:
                st.image(f"{IMAGE_FOLDER}/{predicted}", caption=f"{predicted} (Confidence: {confidence:.2f})", use_container_width=True)
                st.markdown("<div class='neon-box'>", unsafe_allow_html=True)
                chosen_index = extract_index(chosen or "0")
                predicted_index = extract_index(predicted)
                st.markdown("#### 🧠 Why the AI chose this:")
                st.markdown("- It resembles items you picked before.")
                st.markdown(f"- Its index ({predicted_index}) is close to your actual choice ({chosen_index}).")
                if confidence > 0.5:
                    st.markdown("- The AI was pretty confident.")
                else:
                    st.markdown("- The AI made its best guess.")
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.success("✅ Simulation complete! Download your data below.")
    df = pd.DataFrame(st.session_state.choices)
    df["correct_prediction"] = df["selection"] == df["predicted"]
    filename = f"{OUTPUT_DIR}/results_{st.session_state.user_id}.csv"
    df.to_csv(filename, index=False)
    st.dataframe(df)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="your_choices.csv", mime="text/csv")
