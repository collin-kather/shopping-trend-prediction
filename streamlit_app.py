import streamlit as st
import random, time, os, uuid, pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ── CONFIG ──────────────────────────────────────────────────────────────────────
NUM_TRIALS          = 40          # total rounds
TRAIN_AFTER         = 30          # first 30 rounds = training phase
ITEMS_PER_TRIAL     = 5           # one shelf row
TOTAL_TIME_LIMIT    = 120         # 2-minute budget for training phase
IMAGE_FOLDER        = "images"    # folder containing 1.jpg … n.jpg
OUTPUT_DIR          = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────────
extract_index = lambda f: int(os.path.splitext(f)[0]) if f.split(".")[0].isdigit() else -1
get_all_images = lambda: sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(".jpg")],
    key=extract_index,
)

def get_trial_images():
    pool = get_all_images()
    if len(pool) < ITEMS_PER_TRIAL:
        st.error(f"Need ≥{ITEMS_PER_TRIAL} images in `{IMAGE_FOLDER}/`.")
        st.stop()
    return random.sample(pool, ITEMS_PER_TRIAL)

def train_model(records):
    rows = [{"index": extract_index(i), "chosen": int(i == r["selection"])}
            for r in records for i in r["options"]]
    df = pd.DataFrame(rows)
    if df["chosen"].sum() == 0:
        return None
    model = RandomForestClassifier(n_estimators=100)
    model.fit(df[["index"]], df["chosen"])
    return model

def predict_item(model, options):
    df = pd.DataFrame({"index": [extract_index(i) for i in options]})
    probs = model.predict_proba(df)[:, 1]
    best = probs.argmax()
    return options[best], probs[best]

# ── PAGE STYLING ────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"] {
          background: #000 !important;
          color: #00ff00 !important;
          font-family: "Courier New", monospace;
      }
      .neon-box   {border:1px solid #00ff00;padding:10px;border-radius:4px;}
      .card       {border:2px dashed #00ff00;border-radius:4px;padding:6px;margin:4px;background:#111;}
      .timer      {font-size:22px;font-weight:bold;margin:4px 0;color:#00ff00;}
      button      {background:#000;border:1px solid #00ff00 !important;color:#00ff00 !important;}
      button:hover{background:#00ff00;color:#000 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── SESSION STATE INIT ──────────────────────────────────────────────────────────
if "idx" not in st.session_state:
    st.session_state.idx           = 0               # current trial (0-based)
    st.session_state.choices       = []              # list of dicts
    st.session_state.uid           = str(uuid.uuid4())[:8]
    st.session_state.timer_start   = None            # set on Begin
    st.session_state.images        = []              # current 5 images
    st.session_state.model         = None
    st.session_state.model_ready   = False
    st.session_state.last_tick_sec = -1              # for 1-s refresh

# ── INTRO / START BUTTON ────────────────────────────────────────────────────────
st.title("💾 90s Grocery-Shelf AI Simulator")

if st.session_state.timer_start is None:
    st.markdown(
        "Welcome to the **retro shelf challenge**. "
        "You have **2 minutes** to pick 30 items. "
        "Click **Begin** to start!"
    )
    if st.button("🚀 Begin"):
        st.session_state.timer_start = time.time()
        st.rerun()
    st.stop()

# ── TIMER ───────────────────────────────────────────────────────────────────────
elapsed   = int(time.time() - st.session_state.timer_start)
time_left = max(TOTAL_TIME_LIMIT - elapsed, 0)
if st.session_state.idx < TRAIN_AFTER:
    st.markdown(f"<div class='timer'>⏳ Time left in training phase: {time_left} s</div>", unsafe_allow_html=True)

# Auto-refresh once per second for live countdown
now_sec = int(time.time())
if now_sec != st.session_state.last_tick_sec:
    st.session_state.last_tick_sec = now_sec
    st.rerun()

# ── MAIN LOOP ───────────────────────────────────────────────────────────────────
if st.session_state.idx < NUM_TRIALS:
    trial_no = st.session_state.idx + 1
    st.subheader(f"Trial {trial_no} / {NUM_TRIALS}")

    # prepare images
    if not st.session_state.images:
        st.session_state.images = get_trial_images()

    # if model not trained yet, train after 30 rounds
    pred_img, pred_conf = None, None
    if trial_no > TRAIN_AFTER:
        if not st.session_state.model_ready:
            st.session_state.model = train_model(st.session_state.choices)
            st.session_state.model_ready = True
        if st.session_state.model:
            pred_img, pred_conf = predict_item(st.session_state.model, st.session_state.images)

    # render shelf row
    chosen = None
    cols = st.columns(ITEMS_PER_TRIAL)
    for i, (col, img) in enumerate(zip(cols, st.session_state.images)):
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(f"{IMAGE_FOLDER}/{img}", use_container_width=True)
            if st.button("Choose", key=f"btn_{trial_no}_{i}"):
                chosen = img
            st.markdown("</div>", unsafe_allow_html=True)

    # record choice (or timeout with None)
    if chosen or (st.session_state.idx < TRAIN_AFTER and time_left == 0):
        record = dict(
            trial       = trial_no,
            options     = st.session_state.images,
            selection   = chosen,
            predicted   = pred_img,
            response_ms = int((time.time() - st.session_state.timer_start) * 1000),
            timestamp   = datetime.utcnow().isoformat(),
            user_id     = st.session_state.uid,
        )
        st.session_state.choices.append(record)

        # reset for next trial
        st.session_state.idx    += 1
        st.session_state.images  = []
        st.rerun()

    # show prediction outcome **only after 30th round**
    if st.session_state.idx >= TRAIN_AFTER and chosen and pred_img:
        st.markdown("---")
        st.markdown("### 🔍 Results of this round")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**You chose:**")
            st.image(f"{IMAGE_FOLDER}/{chosen}", use_container_width=True)
        with colB:
            st.markdown("**AI predicted:**")
            st.image(f"{IMAGE_FOLDER}/{pred_img}", caption=f"Confidence {pred_conf:.2f}", use_container_width=True)
            st.markdown(
                "<div class='neon-box'>"
                "<u>Why?</u><br>"
                "• Most similar index to your past picks.<br>"
                "• Learned preference from first 30 rounds."
                "</div>",
                unsafe_allow_html=True,
            )

# ── END OF GAME ────────────────────────────────────────────────────────────────
else:
    st.success("Simulation complete! Download your data below.")
    df = pd.DataFrame(st.session_state.choices)
    df["correct"] = df["selection"] == df["predicted"]
    fname = f"{OUTPUT_DIR}/results_{st.session_state.uid}.csv"
    df.to_csv(fname, index=False)
    st.dataframe(df)
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="choices.csv", mime="text/csv")
