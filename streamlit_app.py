import streamlit as st
import random, time, os, uuid, pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ── CONFIG ──────────────────────────────────────────────────────────────────────
NUM_TRIALS       = 40         # total rounds
TRAIN_AFTER      = 30         # first 30 rounds = training phase
ITEMS_PER_TRIAL  = 5          # images per shelf row
TOTAL_TIME_LIMIT = 120        # 2-min budget for training phase
IMAGE_FOLDER     = "images"   # put 1.jpg … n.jpg here
OUTPUT_DIR       = "responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── UTILITIES ───────────────────────────────────────────────────────────────────
def idx(name):  # extract numeric index from filename
    return int(os.path.splitext(name)[0]) if name.split(".")[0].isdigit() else -1

get_all_images  = lambda: sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(".jpg")], key=idx
)
def get_trial_images():
    pool = get_all_images()
    if len(pool) < ITEMS_PER_TRIAL:
        st.error(f"Need ≥{ITEMS_PER_TRIAL} JPG files in “{IMAGE_FOLDER}/”.")
        st.stop()
    return random.sample(pool, ITEMS_PER_TRIAL)

def train(records):
    rows = [{"index": idx(opt), "chosen": int(opt == r["selection"])}
            for r in records for opt in r["options"]]
    df = pd.DataFrame(rows)
    if df["chosen"].sum() == 0:
        return None
    m = RandomForestClassifier(n_estimators=100)
    m.fit(df[["index"]], df["chosen"])
    return m

def predict(m, opts):
    df = pd.DataFrame({"index": [idx(o) for o in opts]})
    p  = m.predict_proba(df)[:, 1]
    best = p.argmax()
    return opts[best], p[best]

# ── STYLING ─────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
      html,body,[class*="css"]{background:#000!important;color:#00ff00!important;
          font-family:"Courier New",monospace;}
      .timer  {font-size:22px;font-weight:bold;margin:6px 0;}
      img.neon{border:2px dashed #00ff00;border-radius:4px;background:#111;padding:3px;}
      button  {background:#000;border:1px solid #00ff00!important;color:#00ff00!important}
      button:hover{background:#00ff00;color:#000!important}
      .neonbox{border:1px solid #00ff00;padding:8px;border-radius:4px;margin-top:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── STATE INIT ──────────────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.update(
        step=0,               # current trial index (0-based)
        picks=[],             # list of dicts
        uid=str(uuid.uuid4())[:8],
        t0=None,              # timer start
        shelf=[],             # current 5 images
        model=None, model_ok=False,
        last_tick=-1,         # for 1-s refresh
        last_res=None         # (chosen, pred, conf)
)

# ── INTRO / START ───────────────────────────────────────────────────────────────
st.title("💾  90s Grocery-Shelf AI Simulator")

if st.session_state.t0 is None:
    st.write(
        "You’ll make **30 speedy picks** (max 2 min). "
        "After that the 90s AI will try to guess your next 10 choices."
    )
    if st.button("🚀  Begin"):
        st.session_state.t0 = time.time()
        st.rerun()
    st.stop()

# ── LIVE TIMER ─────────────────────────────────────────────────────────────────
elapsed   = int(time.time() - st.session_state.t0)
left      = max(TOTAL_TIME_LIMIT - elapsed, 0)
if st.session_state.step < TRAIN_AFTER:
    st.markdown(f"<div class='timer'>⏳ {left} s left in training phase</div>",
                unsafe_allow_html=True)

# Auto-refresh exactly once per second
now = int(time.time())
if now != st.session_state.last_tick:
    st.session_state.last_tick = now
    st.rerun()

# ── MAIN TRIAL LOOP ────────────────────────────────────────────────────────────
if st.session_state.step < NUM_TRIALS:
    n = st.session_state.step + 1
    st.subheader(f"Trial {n} / {NUM_TRIALS}")

    # prepare shelf
    if not st.session_state.shelf:
        st.session_state.shelf = get_trial_images()

    # prediction (only after 30 rounds)
    pred, conf = None, None
    if n > TRAIN_AFTER:
        if not st.session_state.model_ok:
            st.session_state.model = train(st.session_state.picks)
            st.session_state.model_ok = True
        if st.session_state.model:
            pred, conf = predict(st.session_state.model, st.session_state.shelf)

    # show 5 products
    cols = st.columns(ITEMS_PER_TRIAL)
    choice = None
    for col, img in zip(cols, st.session_state.shelf):
        with col:
            st.image(f"{IMAGE_FOLDER}/{img}", use_column_width=True, caption=None,
                     output_format="JPEG", clamp=True, channels="RGB", class_="neon")
            if st.button("Choose", key=f"{n}_{img}"):
                choice = img

    # record on click or if training timer expires with no click
    if choice or (st.session_state.step < TRAIN_AFTER and left == 0):
        st.session_state.picks.append(
            dict(trial=n, options=st.session_state.shelf,
                 selection=choice, predicted=pred,
                 t_ms=int((time.time()-st.session_state.t0)*1000),
                 ts=datetime.utcnow().isoformat(), uid=st.session_state.uid)
        )
        st.session_state.last_res = (choice, pred, conf)  # for later display
        st.session_state.step    += 1
        st.session_state.shelf    = []
        st.rerun()

    # show result ONLY after training phase
    if n > TRAIN_AFTER and choice and pred:
        ch, pr, cf = st.session_state.last_res
        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("### You chose:")
            st.image(f"{IMAGE_FOLDER}/{ch}", use_column_width=True, class_="neon")
        with colB:
            st.markdown("### AI predicted:")
            st.image(f"{IMAGE_FOLDER}/{pr}", caption=f"Conf {cf:.2f}",
                     use_column_width=True, class_="neon")
            st.markdown(
                "<div class='neonbox'>"
                "<u>Why?</u><br/>"
                "• Based on your earlier picks.<br/>"
                "• Close index similarity.<br/>"
                "</div>",
                unsafe_allow_html=True,
            )

# ── FINISH ─────────────────────────────────────────────────────────────────────
else:
    st.success("Simulation complete – download your data!")
    df = pd.DataFrame(st.session_state.picks)
    df["correct"] = df["selection"] == df["predicted"]
    file = f"{OUTPUT_DIR}/choices_{st.session_state.uid}.csv"
    df.to_csv(file, index=False)
    st.dataframe(df)
    st.download_button("📥 CSV", df.to_csv(index=False),
                       file_name="choices.csv", mime="text/csv")
