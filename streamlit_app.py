import streamlit as st
from streamlit_autorefresh import st_autorefresh
import random, time, os, uuid, pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ── CONFIG ──────────────────────────────────────────────────────────────────────
NUM_TRIALS       = 40
TRAIN_AFTER      = 30
ITEMS_PER_TRIAL  = 5
TOTAL_TIME_LIMIT = 120      # two-minute budget for first 30 rounds
IMAGE_FOLDER     = "images"
OUTPUT_DIR       = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────────
extract = lambda f: int(os.path.splitext(f)[0]) if f.split(".")[0].isdigit() else -1
all_imgs = lambda: sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(".jpg")], key=extract)

def trial_images():
    pool = all_imgs()
    if len(pool) < ITEMS_PER_TRIAL:
        st.error(f"Need ≥{ITEMS_PER_TRIAL} images in `{IMAGE_FOLDER}/`.")
        st.stop()
    return random.sample(pool, ITEMS_PER_TRIAL)

def train_model(picks):
    rows = [{"idx": extract(opt), "ch": int(opt == r["sel"])}
            for r in picks for opt in r["opts"]]
    df = pd.DataFrame(rows)
    if df["ch"].sum() == 0:  # no positive labels yet
        return None
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(df[["idx"]], df["ch"])
    return clf

def predict(clf, opts):
    df = pd.DataFrame({"idx": [extract(o) for o in opts]})
    probs = clf.predict_proba(df)[:, 1]
    best  = probs.argmax()
    return opts[best], probs[best]

# ── GLOBAL CSS ──────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
      html,body,[class*="css"]{background:#000!important;color:#00ff00!important;
          font-family:"Courier New",monospace;}
      img.neon{border:2px dashed #00ff00;border-radius:4px;background:#111;padding:3px;}
      button  {background:#000;border:1px solid #00ff00!important;color:#00ff00!important}
      button:hover{background:#00ff00;color:#000!important}
      .neonbox{border:1px solid #00ff00;padding:8px;border-radius:4px;margin-top:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── SESSION STATE ───────────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.update(
        step=0, picks=[], shelf=[],  uid=str(uuid.uuid4())[:8],
        start=None, model=None, model_ready=False, last_res=None
    )

# ── AUTO-REFRESH (1 s) ─────────────────────────────────────────────────────────
st_autorefresh(interval=1_000, limit=None, key="timer-tick")

# ── HEADER + SIDEBAR TIMER ─────────────────────────────────────────────────────
st.title("💾  90s Grocery-Shelf AI Simulator")

if st.session_state.start is None:
    st.write(
        "Pick **30 items** (within 2 minutes) to teach the retro AI. "
        "Then it will try to predict your next 10 choices."
    )
    if st.button("🚀  Begin"):
        st.session_state.start = time.time()
        st.experimental_rerun()
    st.stop()

elapsed   = int(time.time() - st.session_state.start)
remaining = max(TOTAL_TIME_LIMIT - elapsed, 0)
phase_txt = f"{remaining}s left in training phase" if st.session_state.step < TRAIN_AFTER else "Prediction phase"
st.sidebar.markdown(f"## ⏳ {phase_txt}")

# ── MAIN LOOP ───────────────────────────────────────────────────────────────────
if st.session_state.step < NUM_TRIALS:
    n = st.session_state.step + 1
    st.subheader(f"Trial {n} / {NUM_TRIALS}")

    # prepare shelf if empty
    if not st.session_state.shelf:
        st.session_state.shelf = trial_images()

    # prediction (post-training)
    pred, conf = None, None
    if n > TRAIN_AFTER:
        if not st.session_state.model_ready:
            st.session_state.model = train_model(st.session_state.picks)
            st.session_state.model_ready = True
        if st.session_state.model:
            pred, conf = predict(st.session_state.model, st.session_state.shelf)

    # display shelf row
    chosen = None
    for col, img in zip(st.columns(ITEMS_PER_TRIAL), st.session_state.shelf):
        with col:
            st.image(f"{IMAGE_FOLDER}/{img}", class_="neon", use_container_width=True)
            if st.button("Choose", key=f"{n}_{img}"):
                chosen = img

    # record on click or timeout (during training)
    if chosen or (st.session_state.step < TRAIN_AFTER and remaining == 0):
        st.session_state.picks.append(
            dict(trial=n, opts=st.session_state.shelf, sel=chosen,
                 pred=pred, ms=int((time.time()-st.session_state.start)*1000),
                 ts=datetime.utcnow().isoformat(), uid=st.session_state.uid)
        )
        st.session_state.last_res = (chosen, pred, conf)  # save for display
        st.session_state.step    += 1
        st.session_state.shelf    = []
        st.experimental_rerun()

    # result panel (only once predictions start)
    if n > TRAIN_AFTER and chosen and pred:
        ch, pr, cf = st.session_state.last_res
        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("### You chose:")
            st.image(f"{IMAGE_FOLDER}/{ch}", class_="neon", use_container_width=True)
        with colB:
            st.markdown("### AI predicted:")
            st.image(f"{IMAGE_FOLDER}/{pr}", caption=f"Confidence {cf:.2f}", class_="neon",
                     use_container_width=True)
            st.markdown(
                "<div class='neonbox'><u>Why?</u><br>"
                "• Learned from your first 30 picks.<br>"
                "• Chose item with highest similarity index.<br>"
                "</div>",
                unsafe_allow_html=True,
            )

# ── DONE ────────────────────────────────────────────────────────────────────────
else:
    st.success("Simulation complete – download your data!")
    df = pd.DataFrame(st.session_state.picks)
    df["correct"] = df["sel"] == df["pred"]
    csv = df.to_csv(index=False)
    st.download_button("📥  CSV", csv, "choices.csv", "text/csv")
    st.dataframe(df)
