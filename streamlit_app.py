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
TIME_LIMIT = 10
IMAGE_FOLDER = "images"
OUTPUT_DIR = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- FEATURE ----------
def extract_index(filename):
    try:
        return int(os.path.splitext(filename)[0])
    except:
        return -1

# ---------- SESSION ----------
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.user_id = str(uuid.uuid4())[:8]
    st.session_state.refresh_images = True
    st.session_state.model_trained = False
    st.session_state.last_result = None
    st.session_state.timer_started = time.time()

# ---------- IMAGE UTILS ----------
def get_all_images():
    return sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")], key=extract_index)

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
    .retail-card {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 8px;
        text-align: center;
        background-color: #fdfdfd;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
    }
    .timer {
        font-size: 20px;
        font-weight: bold;
        color: #aa4400;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("🛒 Grocery Shelf Selection Game")

# ---------- SHOW LAST RESULT ----------
if st.session_state.last_result:
    chosen, predicted, confidence = st.session_state.last_result

    st.markdown("### 🧾 Previous Round Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("🛍️ **You chose:**")
        if chosen:
            st.image(f"{IMAGE_FOLDER}/{chosen}", caption=chosen, use_container_width=True)
        else:
            st.markdown("_No selection made (timed out)_")

    with col2:
        st.markdown("🤖 **We predicted:**")
        if predicted:
            st.image(f"{IMAGE_FOLDER}/{predicted}", caption=f"{predicted} (Confidence: {confidence:.2f})", use_container_width=True)

            # Fun fake explanations
            chosen_index = extract_index(chosen or "0")
            predicted_index = extract_index(predicted)
            distance = abs(predicted_index - chosen_index)

            st.markdown("#### 🧠 Why the model chose this:")
            st.markdown("- It looks similar to items you've picked before.")
            st.markdown(f"- Its index number ({predicted_index}) is close to your actual choice ({chosen_index}).")
            if confidence > 0.5:
                st.markdown("- The model was pretty confident about this guess!")
            else:
                st.markdown("- The model wasn't sure, but this was its best bet.")

# ---------- MAIN TRIAL ----------
if st.session_state.trial_index < NUM_TRIALS:
    trial_num = st.session_state.trial_index + 1
    st.markdown(f"### Trial {trial_num} of {NUM_TRIALS}")

    elapsed = int(time.time() - st.session_state.timer_started)
    remaining_time = max(TIME_LIMIT - elapsed, 0)
    st.markdown(f"⏳ <span class='timer'>Time left: {remaining_time} seconds</span>", unsafe_allow_html=True)

    if st.session_state.refresh_images:
        st.session_state.current_images = get_trial_images()
        st.session_state.refresh_images = False
        st.session_state.timer_started = time.time()

    # Model prediction
    predicted_img, pred_score = None, None
    if trial_num > TRAIN_AFTER:
        if not st.session_state.model_trained:
            st.session_state.model = train_model(st.session_state.choices)
            st.session_state.model_trained = True
        if st.session_state.model:
            predicted_img, pred_score = predict_choice(st.session_state.model, st.session_state.current_images)

    # Display shelf row
    cols = st.columns(ITEMS_PER_TRIAL)
    selected = None
    for i, img in enumerate(st.session_state.current_images):
        with cols[i]:
            st.markdown(f"<div class='retail-card'>", unsafe_allow_html=True)
            st.image(f"{IMAGE_FOLDER}/{img}", use_container_width=True)
            if st.button("Choose", key=f"choose_{i}"):
                selected = img
            st.markdown("</div>", unsafe_allow_html=True)

    # Handle timeout or selection
    if remaining_time <= 0 or selected:
        final_selection = selected if selected else None
        st.session_state.choices.append({
            "trial": trial_num,
            "options": st.session_state.current_images,
            "selection": final_selection,
            "predicted": predicted_img,
            "response_time": elapsed,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": st.session_state.user_id
        })

        st.session_state.last_result = (final_selection, predicted_img, pred_score)
        st.session_state.trial_index += 1
        st.session_state.refresh_images = True
        st.experimental_rerun()
    else:
        st.experimental_rerun()

# ---------- END ----------
else:
    st.success("✅ You're done! Here are your results:")
    df = pd.DataFrame(st.session_state.choices)
    df["correct_prediction"] = df["selection"] == df["predicted"]
    filename = f"{OUTPUT_DIR}/results_{st.session_state.user_id}.csv"
    df.to_csv(filename, index=False)
    st.dataframe(df)
    st.download_button("Download Results CSV", data=df.to_csv(index=False), file_name="your_choices.csv", mime="text/csv")
