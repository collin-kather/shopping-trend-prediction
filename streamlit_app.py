import streamlit as st
import pandas as pd
import random
import os
import uuid
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# --------- CONFIG ---------
NUM_TRIALS = 40
TRAIN_AFTER = 30
ITEMS_PER_TRIAL = 5
IMAGE_FOLDER = "images"
FEATURE_CSV = "simple_image_features.csv"
OUTPUT_DIR = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- HELPER FUNCTIONS ---------
def extract_index(filename):
    try:
        return int(os.path.splitext(filename)[0])
    except:
        return -1

def load_features():
    df = pd.read_csv(FEATURE_CSV)
    df["index"] = df["image_name"].apply(lambda x: int(os.path.splitext(x)[0]))
    return df

def get_all_images():
    return sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")], key=extract_index)

def get_trial_images():
    all_images = get_all_images()
    return random.sample(all_images, ITEMS_PER_TRIAL)

def train_model(choices, feature_df):
    rows = []
    for trial in choices:
        for img in trial["options"]:
            idx = extract_index(img)
            row = feature_df[feature_df["index"] == idx].copy()
            if not row.empty:
                row["chosen"] = int(img == trial["selection"])
                rows.append(row)
    if not rows:
        return None
    df = pd.concat(rows)
    if df["chosen"].sum() == 0:
        return None
    X = df.drop(columns=["image_name", "index", "chosen"])
    y = df["chosen"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def predict_choice(model, options, feature_df):
    rows = []
    for img in options:
        idx = extract_index(img)
        row = feature_df[feature_df["index"] == idx].copy()
        if not row.empty:
            row["image_name"] = img
            rows.append(row)
    df = pd.concat(rows)
    X = df.drop(columns=["image_name", "index"])
    probs = model.predict_proba(X)[:, 1]
    best_idx = probs.argmax()
    return df.iloc[best_idx]["image_name"], probs[best_idx]

# --------- STYLING ---------
st.set_page_config(layout="wide")
st.markdown("""
<style>
body { background-color: #101010; color: #00FF66; }
.retail-card {
    background-color: #111;
    padding: 8px;
    border: 2px solid #00FF66;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 15px;
}
h1, h2, h3 { color: #00FF66; }
button { background-color: #222; color: #00FF66; }
</style>
""", unsafe_allow_html=True)

# --------- SESSION INIT ---------
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.user_id = str(uuid.uuid4())[:8]
    st.session_state.model_trained = False
    st.session_state.last_result = None

# --------- LOAD FEATURES ---------
feature_df = load_features()

# --------- HEADER ---------
st.title("🛒 RETRO CEREAL SELECTION SIM")

# --------- DISPLAY RESULT ---------
if st.session_state.last_result and st.session_state.trial_index > TRAIN_AFTER:
    chosen, predicted, confidence = st.session_state.last_result
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛍️ You Chose:")
        img_path = os.path.join(IMAGE_FOLDER, chosen)
        if os.path.exists(img_path):
            st.image(img_path, width=200)
        else:
            st.error(f"Couldn't find {img_path}")
    with col2:
        st.markdown(f"### 🤖 AI Predicted:")
        pred_path = os.path.join(IMAGE_FOLDER, predicted)
        if os.path.exists(pred_path):
            st.image(pred_path, width=200)
            st.markdown(f"- Confidence: `{confidence:.2f}`")
        else:
            st.error(f"Couldn't find {pred_path}")

# --------- MAIN GAME LOOP ---------
if st.session_state.trial_index < NUM_TRIALS:
    trial_num = st.session_state.trial_index + 1
    st.markdown(f"## Trial {trial_num} of {NUM_TRIALS}")
    current_images = get_trial_images()

    # Predict
    predicted_img, pred_score = None, None
    if trial_num > TRAIN_AFTER and not st.session_state.model_trained:
        model = train_model(st.session_state.choices, feature_df)
        if model:
            st.session_state.model = model
            st.session_state.model_trained = True

    if trial_num > TRAIN_AFTER and st.session_state.model_trained:
        predicted_img, pred_score = predict_choice(st.session_state.model, current_images, feature_df)

    cols = st.columns(ITEMS_PER_TRIAL)
    selected = None
    for i, img in enumerate(current_images):
        with cols[i]:
            st.markdown(f"<div class='retail-card'>", unsafe_allow_html=True)
            img_path = os.path.join(IMAGE_FOLDER, img)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            if st.button("Choose", key=f"choose_{i}_{trial_num}"):
                selected = img
            st.markdown("</div>", unsafe_allow_html=True)

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
        st.experimental_rerun()

else:
    st.success("✅ Game Complete! Download your results:")
    df = pd.DataFrame(st.session_state.choices)
    df.to_csv(f"{OUTPUT_DIR}/results_{st.session_state.user_id}.csv", index=False)
    st.dataframe(df)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="results.csv")
