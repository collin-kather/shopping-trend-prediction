import streamlit as st
import os
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- CONFIG ---
IMAGE_FOLDER = "images"
CSV_FILE = "simple_image_features.csv"
NUM_TRIALS = 40
TRAIN_AFTER = 30
ITEMS_PER_TRIAL = 5

# --- LOAD FEATURES ---
@st.cache_data
def load_features():
    df = pd.read_csv(CSV_FILE)
    df["index"] = df["filename"].apply(lambda x: int(os.path.splitext(x)[0]))
    return df.set_index("index")

feature_df = load_features()

# --- SESSION STATE ---
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
    st.session_state.choices = []
    st.session_state.model = None

# --- HEADER ---
st.title("🛒 Grocery Shelf Selector (Smarter AI Edition)")

# --- MAIN LOGIC ---
if st.session_state.trial_index < NUM_TRIALS:
    st.subheader(f"Trial {st.session_state.trial_index + 1} of {NUM_TRIALS}")

    available_images = feature_df.index.tolist()
    selected_images = random.sample(available_images, ITEMS_PER_TRIAL)

    cols = st.columns(ITEMS_PER_TRIAL)
    user_choice = None

    for i, img_index in enumerate(selected_images):
        img_file = feature_df.loc[img_index]["filename"]
        with cols[i]:
            st.image(os.path.join(IMAGE_FOLDER, img_file), use_container_width=True)
            if st.button("Choose", key=f"choose_{i}"):
                user_choice = img_index

    if user_choice is not None:
        choice_record = {
            "trial": st.session_state.trial_index,
            "options": selected_images,
            "selected": user_choice
        }

        # Train model if enough data
        if st.session_state.trial_index >= TRAIN_AFTER:
            train_data = []
            for trial in st.session_state.choices:
                for img in trial["options"]:
                    train_data.append({
                        "index": img,
                        "PC1": feature_df.loc[img]["PC1"],
                        "PC2": feature_df.loc[img]["PC2"],
                        "PC3": feature_df.loc[img]["PC3"],
                        "chosen": int(img == trial["selected"])
                    })
            df_train = pd.DataFrame(train_data)
            model = RandomForestClassifier(n_estimators=100)
            model.fit(df_train[["PC1", "PC2", "PC3"]], df_train["chosen"])
            st.session_state.model = model

            # Predict on new trial
            X_test = feature_df.loc[selected_images][["PC1", "PC2", "PC3"]]
            preds = model.predict_proba(X_test)[:, 1]
            best_idx = np.argmax(preds)
            predicted = selected_images[best_idx]
            choice_record["predicted"] = predicted
            choice_record["confidence"] = preds[best_idx]

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.success("🛍️ You chose:")
                st.image(os.path.join(IMAGE_FOLDER, feature_df.loc[user_choice]["filename"]), width=200)
            with col2:
                st.info("🤖 We predicted:")
                st.image(os.path.join(IMAGE_FOLDER, feature_df.loc[predicted]["filename"]), width=200)
                st.caption(f"Confidence: {preds[best_idx]:.2f}")

        st.session_state.choices.append(choice_record)
        st.session_state.trial_index += 1
        st.experimental_rerun()

else:
    st.success("✅ You finished all trials!")
    df_results = pd.DataFrame(st.session_state.choices)
    st.dataframe(df_results)
    st.download_button("Download results", df_results.to_csv(index=False), file_name="results.csv")
