import streamlit as st
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- Funkcje z projektu (minimalny zestaw) ----------
def load_league_logos(base_dir="herby"):
    base = Path(base_dir)
    data = {}
    for league_dir in base.iterdir():
        if league_dir.is_dir():
            league = league_dir.name
            data[league] = []
            for p in league_dir.glob("*.png"):
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    data[league].append((p, img))
    return data

def make_binary_mask(img):
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        return mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def shape_features_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    circ = 0.0 if area == 0 or per == 0 else float(4*np.pi*area/(per*per))
    x,y,w,h = cv2.boundingRect(cnt)
    aspect = float(w/h) if h else 0.0
    size = float(np.count_nonzero(mask))
    return circ, aspect, size, cnt, (x,y,w,h)

def extract_row(path, img, league):
    mask = make_binary_mask(img)
    out = shape_features_from_mask(mask)
    if out is None:
        return None
    circ, aspect, size, cnt, bbox = out
    return {
        "league": league,
        "path": str(path),
        "circularity": circ,
        "aspect_ratio": aspect,
        "size_pixels": size
    }

@st.cache_data
def build_df(base_dir="herby"):
    logos = load_league_logos(base_dir)
    rows = []
    for league, items in logos.items():
        for p, img in items:
            r = extract_row(p, img, league)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)

def plot_violin_box_points(df, leagues):
    data = [df[df.league==lg]["circularity"].values for lg in leagues]
    fig, ax = plt.subplots(figsize=(10,5))

    vp = ax.violinplot(data, showextrema=False)
    for b in vp["bodies"]:
        b.set_alpha(0.15)

    bp = ax.boxplot(data, tick_labels=leagues, patch_artist=True, whis=(2.5,97.5))
    for m in bp["medians"]:
        m.set_color("red"); m.set_linewidth(2.5)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(i, 0.06, size=len(vals))
        ax.scatter(x, vals, s=18, alpha=0.35)

    means = [np.mean(v) for v in data]
    ax.scatter(range(1,len(leagues)+1), means, marker="D", s=60, color="black", label="Średnia")

    ax.set_ylim(0,1)
    ax.set_title("Kolistość herbów – TOP5")
    ax.set_ylabel("Kolistość (0–1)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend()
    st.pyplot(fig)

def show_logo_panel(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    mask = make_binary_mask(img)
    out = shape_features_from_mask(mask)
    if out is None:
        st.warning("Nie udało się wyznaczyć konturu.")
        return
    circ, aspect, size, cnt, (x,y,w,h) = out

    bgr = img[:,:,:3].copy()
    overlay = bgr.copy()
    cv2.drawContours(overlay, [cnt], -1, (0,255,0), 2)
    bbox_img = bgr.copy()
    cv2.rectangle(bbox_img, (x,y), (x+w, y+h), (255,0,0), 2)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    def to_rgb(a): return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    c1, c2, c3 = st.columns(3)
    c1.image(to_rgb(bgr), caption="Oryginał", width=True)
    c2.image(mask, caption="Maska", width=True)
    c3.image(to_rgb(overlay), caption="Kontur", width=True)

    c4, c5, c6 = st.columns(3)
    c4.image(to_rgb(bbox_img), caption="Bounding box", width=True)
    c5.image(edges, caption="Krawędzie (Canny)", width=True)
    c6.metric("Kolistość", f"{circ:.3f}")
    st.caption(f"aspect_ratio={aspect:.3f}, size_pixels={size:.0f}")

# ---------- UI ----------
st.set_page_config(page_title="ADM – Herby TOP5", layout="wide")
st.title("Dashboard: analiza herbów lig TOP5")

df = build_df("C:/Users/konra/PycharmProjects/adm1/adm_logos/herby")
if df.empty:
    st.error("Brak danych. Sprawdź folder herby/ (eng/esp/fra/ger/ita).")
    st.stop()

leagues = sorted(df["league"].unique())

with st.sidebar:
    st.header("Filtry")
    chosen = st.multiselect("Ligi", leagues, default=leagues)
    view_league = st.selectbox("Podgląd herbu (liga)", leagues)
    mode = st.radio("Ekstrema", ["Najbardziej okrągłe", "Najmniej okrągłe"])

st.subheader("Rozkład kolistości")
plot_violin_box_points(df[df["league"].isin(chosen)], sorted(chosen))

st.divider()
st.subheader("Podgląd herbów + diagnostyka (różne „kąty”)")
sub = df[df["league"]==view_league].copy()
sub = sub.sort_values("circularity", ascending=(mode=="Najmniej okrągłe")).head(12)

cols = st.columns(6)
for i, row in enumerate(sub.itertuples(), start=0):
    img = cv2.imread(row.path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB)
    with cols[i % 6]:
        if st.button(f"{row.circularity:.2f}", key=row.path):
            st.session_state["selected_path"] = row.path
        st.image(rgb, width=True)

selected = st.session_state.get("selected_path", sub.iloc[0]["path"])
st.write("Wybrany plik:", selected)
show_logo_panel(selected)
