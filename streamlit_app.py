import streamlit as st, random, time, os, uuid, pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ── BASIC CONFIG ────────────────────────────────────────────────────────────────
ROUNDS            = 40       # total trials
TRAIN_ROUNDS      = 30       # first 30 train the model
CHOICES_PER_ROW   = 5        # images shown per round
TIME_PER_ROUND    = 10       # seconds per round
IMG_FOLDER        = "images"
OUT_DIR           = "responses"
os.makedirs(OUT_DIR, exist_ok=True)

# ── SMALL HELPERS ───────────────────────────────────────────────────────────────
num = lambda f: int(os.path.splitext(f)[0]) if f.split(".")[0].isdigit() else -1
all_imgs = lambda: sorted([f for f in os.listdir(IMG_FOLDER) if f.endswith(".jpg")], key=num)
def shelf(): return random.sample(all_imgs(), CHOICES_PER_ROW)

def train(log):
    rows = [{"idx": num(i), "y": int(i == r["sel"])} for r in log for i in r["opts"]]
    df = pd.DataFrame(rows)
    if df["y"].sum() == 0: return None
    m = RandomForestClassifier(); m.fit(df[["idx"]], df["y"]); return m
def predict(m, opts):
    import numpy as np
    p = m.predict_proba(pd.DataFrame({"idx": [num(o) for o in opts]}))[:, 1]
    j = int(np.argmax(p)); return opts[j], p[j]

# ── STATE ───────────────────────────────────────────────────────────────────────
if "round" not in st.session_state:
    st.session_state.update(
        round=0, log=[], images=[], uid=uuid.uuid4().hex[:8],
        start=time.time(), model=None, trained=False, last=None
    )

# ── PRETTY CSS ──────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.markdown("""
<style>
/* -------------  background + fonts ------------- */
html, body, [class*="css"] {
    background: radial-gradient(circle at top left,#242424 0%,#000 80%) no-repeat;
    color:#fafafa; font-family: "Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
/* -------------  header accent ------------- */
h1 {color:#FDE047;text-shadow:0 0 4px #fff;}
/* -------------  image card ------------- */
div.card {
    background:#111;border-radius:10px;padding:8px;
    transition:transform .15s,box-shadow .15s;
}
div.card:hover {transform:translateY(-4px);box-shadow:0 4px 12px rgba(0,0,0,.4);}
img {border-radius:6px;}
/* -------------  buttons ------------- */
button {border:none;border-radius:6px;padding:4px 12px;
        background:#FDE047;color:#000;font-weight:600;}
button:hover {background:#ffe95c;}
/* -------------  timer ------------- */
.tag {background:#444;padding:4px 8px;border-radius:4px;font-size:13px;}
.tag.crit {background:#c4302b;color:#fff;}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────────────────
st.title("🛍️  Grocery-Shelf Choice Game")

# ── TIMER TAG ───────────────────────────────────────────────────────────────────
elapsed = int(time.time() - st.session_state.start)
remain  = max(TIME_PER_ROUND - (elapsed % TIME_PER_ROUND), 0)
timer_class = "crit" if remain <= 3 else ""
st.markdown(f"<span class='tag {timer_class}'>⏱ {remain}s left</span>", unsafe_allow_html=True)

# ── LAST RESULT (only after training) ───────────────────────────────────────────
if st.session_state.last and st.session_state.round > TRAIN_ROUNDS:
    ch, pr, cf = st.session_state.last
    st.markdown("#### Last round")
    col1, col2 = st.columns(2)
    with col1:  st.markdown("You chose:"); st.image(f"{IMG_FOLDER}/{ch}", width=120)
    with col2:  st.markdown(f"AI guessed ({cf:.2f}):"); st.image(f"{IMG_FOLDER}/{pr}", width=120)

st.divider()

# ── MAIN GAME ───────────────────────────────────────────────────────────────────
if st.session_state.round < ROUNDS:

    r = st.session_state.round + 1
    st.subheader(f"Round {r}/{ROUNDS}")

    # build shelf once
    if not st.session_state.images:
        st.session_state.images = shelf()

    # possible prediction
    pred_img = pred_conf = None
    if r > TRAIN_ROUNDS:
        if not st.session_state.trained:
            st.session_state.model = train(st.session_state.log)
            st.session_state.trained = True
        if st.session_state.model:
            pred_img, pred_conf = predict(st.session_state.model, st.session_state.images)

    # show images
    choice = None
    for col, img in zip(st.columns(CHOICES_PER_ROW), st.session_state.images):
        with col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(f"{IMG_FOLDER}/{img}", use_container_width=True)
            if st.button("Choose", key=f"{r}_{img}"): choice = img
            st.markdown("</div>", unsafe_allow_html=True)

    # auto-next after TIME_PER_ROUND or on click
    time_up = (elapsed // TIME_PER_ROUND) >= r
    if choice or time_up:
        st.session_state.log.append(dict(round=r, opts=st.session_state.images,
                    sel=choice, pred=pred_img, conf=pred_conf))
        st.session_state.last   = (choice, pred_img, pred_conf)
        st.session_state.round += 1
        st.session_state.images = []
        st.experimental_rerun()

# ── RESULTS ────────────────────────────────────────────────────────────────────
else:
    st.success("Done!  Download your results:")
    df = pd.DataFrame(st.session_state.log)
    df["correct"] = df["sel"] == df["pred"]
    st.download_button("⬇️ CSV", df.to_csv(index=False), "choices.csv", "text/csv")
    st.dataframe(df)
