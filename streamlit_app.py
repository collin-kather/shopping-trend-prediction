import streamlit as st, random, time, os, uuid, pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ── CONFIG ──────────────────────────────────────────────────────────────────────
TOTAL_ROUNDS      = 40
TRAIN_ROUNDS      = 30          # first 30 = training
OPTIONS_PER_ROUND = 5
TRAIN_DEADLINE    = 120         # seconds to finish first 30 (soft limit)
IMG_FOLDER        = "images"
OUT_FOLDER        = "responses"
os.makedirs(OUT_FOLDER, exist_ok=True)

# ── UTILITIES ───────────────────────────────────────────────────────────────────
def idx(fname):          # numeric part of “123.jpg”
    return int(os.path.splitext(fname)[0]) if fname.split(".")[0].isdigit() else -1

def pool():
    return sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(".jpg")], key=idx)

def new_shelf():
    all_imgs = pool()
    if len(all_imgs) < OPTIONS_PER_ROUND:
        st.error(f"Need ≥{OPTIONS_PER_ROUND} .jpg files in `{IMG_FOLDER}/`"); st.stop()
    return random.sample(all_imgs, OPTIONS_PER_ROUND)

def train_model(records):
    rows = [{"i": idx(opt), "y": int(opt == r["sel"])}
            for r in records for opt in r["opts"]]
    df = pd.DataFrame(rows)
    if df["y"].sum() == 0: return None
    clf = RandomForestClassifier(n_estimators=100); clf.fit(df[["i"]], df["y"]); return clf

def predict(clf, opts):
    import numpy as np
    probs = clf.predict_proba(pd.DataFrame({"i": [idx(o) for o in opts]}))[:, 1]
    best  = int(np.argmax(probs))
    return opts[best], probs[best]

# ── UI + STATE ──────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🛒 Minimal Grocery-Shelf AI Demo")

if "state" not in st.session_state:
    st.session_state.state  = "intro"
    st.session_state.step   = 0
    st.session_state.log    = []
    st.session_state.shelf  = []
    st.session_state.uid    = uuid.uuid4().hex[:8]
    st.session_state.start  = None
    st.session_state.model  = None
    st.session_state.ready  = False
    st.session_state.last   = None

# ---------- INTRO ----------
if st.session_state.state == "intro":
    st.write(
        f"Pick **{TRAIN_ROUNDS} items** (try to stay under 2 min). "
        "After that the AI will predict your next choices."
    )
    if st.button("Start"):
        st.session_state.state  = "game"
        st.session_state.start  = time.time()
        st.experimental_rerun()
    st.stop()

# ---------- TIMER / STATUS ----------
elapsed = int(time.time() - st.session_state.start)
time_left = max(TRAIN_DEADLINE - elapsed, 0)
phase = "Training" if st.session_state.step < TRAIN_ROUNDS else "Prediction"
st.write(f"⏱ **{phase} phase** – {time_left}s left (soft limit)")

# ---------- GAME LOOP ----------
if st.session_state.step < TOTAL_ROUNDS:

    round_no = st.session_state.step + 1
    st.subheader(f"Round {round_no} / {TOTAL_ROUNDS}")

    # load shelf once
    if not st.session_state.shelf:
        st.session_state.shelf = new_shelf()

    # maybe produce a prediction
    pred_img, conf = None, None
    if round_no > TRAIN_ROUNDS:
        if not st.session_state.ready:
            st.session_state.model = train_model(st.session_state.log)
            st.session_state.ready = True
        if st.session_state.model:
            pred_img, conf = predict(st.session_state.model, st.session_state.shelf)

    # display 5 options
    choice = None
    for col, img in zip(st.columns(OPTIONS_PER_ROUND), st.session_state.shelf):
        with col:
            st.image(f"{IMG_FOLDER}/{img}", width=180)
            if st.button("Choose", key=f"{round_no}_{img}"):
                choice = img

    # record click (or skip if deadline passed)
    if choice or (st.session_state.step < TRAIN_ROUNDS and time_left == 0):
        st.session_state.log.append(
            dict(r=round_no, opts=st.session_state.shelf,
                 sel=choice, pred=pred_img, conf=conf,
                 ts=datetime.utcnow().isoformat())
        )
        st.session_state.last  = (choice, pred_img, conf)
        st.session_state.step += 1
        st.session_state.shelf = []
        st.experimental_rerun()

    # show immediate result only in prediction phase
    if round_no > TRAIN_ROUNDS and choice and pred_img:
        ch, pr, cf = st.session_state.last
        st.markdown("**You picked:**")
        st.image(f"{IMG_FOLDER}/{ch}", width=150)
        st.markdown("**AI guessed:**")
        st.image(f"{IMG_FOLDER}/{pr}", caption=f"Conf {cf:.2f}", width=150)

# ---------- DONE ----------
else:
    st.success("All rounds complete!")
    df = pd.DataFrame(st.session_state.log)
    df["correct"] = df["sel"] == df["pred"]
    st.download_button("📥 Download CSV", df.to_csv(index=False), "choices.csv", "text/csv")
    st.dataframe(df)
