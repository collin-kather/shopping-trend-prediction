import streamlit as st
import pandas as pd
import random
import os
import uuid
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ---------- CONFIG ----------
NUM_TRIALS = 40
TRAIN_AFTER = 30
ITEMS_PER_TRIAL = 5
IMAGE_FOLDER = "images"
FEATURE_CSV = "simple_image_features.csv"
OUTPUT_DIR = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- SESSION STATE ----------
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.user_id = str(uuid.uuid4())[:8]
    st.session_state.model_trained = False
    st.session_state.last_result = None

# ---------- STYLING ----------
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    body {
        background-color: black;
    }
    .retail-card {
        border: 2px solid #00ff00;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 8px;
        text-align: center;
        background-color: #111;
        box-shadow: 1px 1px 5px rgba(0,255,0,0.2);
    }
    .timer {
        font-size: 20px;
        font-weight: bold;
        color: #00ff00;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD FEATURES ----------
@st.cache_data
def load_feature_csv():
    df = pd.read_csv(FEATURE_CSV)
    return df.set_index("filename")

features_df = load_feature_csv()

# ---------- ML UTILS ----------
def train_model(data, features_df):
    records = []
    for trial in data:
        for img in trial["options"]:
            if img in features_df.index:
                feat = features_df.loc[img].values
                records.append({"features": feat, "chosen": int(img == trial["selection"])})

    if not records:
        return None

    X = [r["features"] for r in records]
    y = [r["chosen"] for r in records]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def predict_choice(model, image_list, features_df):
    X_pred = [features_df.loc[img].values for img in image_list if img in features_df.index]
    preds = model.predict_proba(X_pred)
    confidences = preds[:, 1]
    best_idx = confidences.argmax()
    return image_list[best_idx], confidences[best_idx]

# ---------- IMAGE UTILS ----------
def get_all_images():
    return sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")])

def get_trial_images():
    all_images = get_all_images()
    return random.sample(all_images, ITEMS_PER_TRIAL)

# ---------- HEADER ----------
st.title("🛒 Cereal Surreal: Grocery Shelf Selection")

# ---------- DISPLAY RESULTS ----------
if st.session_state.last_result:
    chosen, predicted, confidence = st.session_state.last_result
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛍️ You chose:")
        st.image(f"{IMAGE_FOLDER}/{chosen}", width=200)
    with col2:
        st.markdown("### 🤖 We predicted:")
        st.image(f"{IMAGE_FOLDER}/{predicted}", width=200)
        st.markdown(f"Confidence: **{confidence:.2f}**")

# ---------- MAIN GAME ----------
if st.session_state.trial_index < NUM_TRIALS:
    trial_num = st.session_state.trial_index + 1
    st.markdown(f"## Trial {trial_num} of {NUM_TRIALS}")

    current_images = get_trial_images()
    selected = None

    # Train model
    model, predicted_img, pred_score = None, None, None
    if trial_num > TRAIN_AFTER:
        if not st.session_state.model_trained:
            model = train_model(st.session_state.choices, features_df)
            st.session_state.model = model
            st.session_state.model_trained = True
        else:
            model = st.session_state.model
        if model:
            predicted_img, pred_score = predict_choice(model, current_images, features_df)

    # Display images
    cols = st.columns(ITEMS_PER_TRIAL)
    for i, img in enumerate(current_images):
        with cols[i]:
            st.markdown(f"<div class='retail-card'>", unsafe_allow_html=True)
            st.image(f"{IMAGE_FOLDER}/{img}", use_container_width=True)
            if st.button("Choose", key=f"btn_{i}"):
                selected = img
            st.markdown("</div>", unsafe_allow_html=True)

    # Handle result
    if selected:
        st.session_state.choices.append({
            "trial": trial_num,
            "options": current_images,
            "selection": selected,
            "predicted": predicted_img,
            "response_time": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": st.session_state.user_id
        })

        st.session_state.last_result = (selected, predicted_img, pred_score)
        st.session_state.trial_index += 1
        st.rerun()

else:
    st.success("✅ You're done! Here are your results:")
    df = pd.DataFrame(st.session_state.choices)
    df["correct"] = df["selection"] == df["predicted"]
    fname = f"{OUTPUT_DIR}/results_{st.session_state.user_id}.csv"
    df.to_csv(fname, index=False)
    st.dataframe(df)
    st.download_button("Download Results", data=df.to_csv(index=False), file_name="your_choices.csv", mime="text/csv")
