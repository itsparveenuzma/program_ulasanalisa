import re
import io
import os
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import altair as alt
import nltk

from google_play_scraper import app as gp_app, reviews, Sort
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix

# PAGE CONFIG
st.set_page_config(
    page_title="UlasAnalisa – Prediksi Sentimen",
    page_icon="static/logo_ulas.png", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# SESSION STATE BOOTSTRAP
for k, v in {
    "results": {},
    "app_id": None,
    "csv_pred": None,
    "csv_dist": None,
    "is_combo": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
@media (max-width: 900px) {
  .block-container .row-widget.stHorizontal {flex-wrap: wrap;}
}
</style>
""", unsafe_allow_html=True)

#LOGO base64
def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_left_b64 = img_to_base64("static/logo_ulas.png")
logo_right_b64 = img_to_base64("static/fti_untar.png")

#BACA PAGE DARI URL
try:  
    page = st.query_params.get("page", "home")
except AttributeError:  
    page = st.experimental_get_query_params().get("page", ["home"])[0]

home_active = "active" if page == "home" else ""
pred_active = "active" if page == "prediksi" else ""
tentang_active = "active" if page == "tentang" else ""

#NAVBAR
st.markdown(
    f"""
<style>
.navbar {{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 80px;
    background: #ffffff;
    display: flex;
    align-items: center;
    padding: 0 1.5rem;
    border-bottom: 3px solid #b71c1c;
    z-index: 999999;
}}
/* turunkan konten & sidebar */
[data-testid="stAppViewContainer"] > .main {{ margin-top: 90px; }}
[data-testid="stSidebar"] {{ top: 90px; }}

/* kiri & kanan sama lebar dan di-center */
.nav-left, .nav-right {{
    width: 220px;
    display: flex;
    justify-content: center;
    align-items: center;
}}

/* menu tepat di tengah */
.nav-center {{
    flex: 1;
    display: flex;
    justify-content: center;
    gap: 2.5rem;
}}

.nav-center a {{
    text-decoration: none;
    color: #444;
    font-weight: 500;
}}
.nav-center a.active {{
    color: #b71c1c;
    border-bottom: 2px solid #b71c1c;
    padding-bottom: 4px;
}}

.logo-left {{ height: 150px; }}
.logo-right {{ height: 65px; }}
</style>

<div class="navbar">
  <div class="nav-left">
    <img src="data:image/png;base64,{logo_left_b64}" class="logo-left">
  </div>
  <div class="nav-center">
     <a href="?page=home" target="_self" class="{home_active}">Beranda</a>
    <a href="?page=prediksi" target="_self" class="{pred_active}">Prediksi</a>
    <a href="?page=tentang" target="_self" class="{tentang_active}">Tentang</a>
  </div>
  <div class="nav-right">
    <img src="data:image/png;base64,{logo_right_b64}" class="logo-right">
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("""
<style>
@media (max-width: 900px) {
  [data-testid="stHeader"]{
    display: flex !important;
    position: fixed;
    top: 90px;           
    left: 0; right: 0;
    background: transparent;
    z-index: 1000000;    
  }

  .navbar{ top: 0; z-index: 999999; }

  [data-testid="stAppViewContainer"] > .main{ margin-top: 140px; } 
  [data-testid="stSidebar"]{ top: 140px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root{ --nav-h: 90px; }   

@media (max-width: 900px){
  [data-testid="stHeader"]{
    display:flex !important;
    position:fixed;
    top:var(--nav-h);
    left:0; right:0;
    background:transparent;
    z-index:1000000;
    height:48px;          
  }

  [data-testid="stSidebar"]{
    top:var(--nav-h) !important;
    height:calc(100vh - var(--nav-h)) !important;
  }

  [data-testid="stSidebar"] .block-container{ padding-top:.5rem !important; }

  [data-testid="stAppViewContainer"] > .main{
    margin-top:calc(var(--nav-h) + 48px) !important;
  }
}
</style>
""", unsafe_allow_html=True)

NLTK_DIR = "/home/appuser/nltk_data" 
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DIR)

@st.cache_resource(show_spinner=False)
def ensure_stopwords():
    """Pastikan korpus 'stopwords' tersedia; unduh bila belum ada.
       Return: set stopwords gabungan id + en dengan fallback lokal jika unduhan gagal."""
    try:
        from nltk.corpus import stopwords as _sw
        _ = _sw.words('english')
    except LookupError:
        try:
            nltk.download('stopwords', download_dir=NLTK_DIR, quiet=True)
        except Exception:
            pass  

    try:
        from nltk.corpus import stopwords as _sw
        id_sw = set(_sw.words('indonesian'))
        en_sw = set(_sw.words('english'))
        return id_sw | en_sw
    except Exception:
        id_fallback = {
            'yang','dan','di','ke','dari','pada','untuk','dengan','atau','tidak','ini',
            'itu','saya','kami','kamu','dia','mereka','akan','juga','karena','sebagai',
            'ada','jadi','agar','dalam','pada','dapat','sudah','belum','lebih'
        }
        en_fallback = {
            'the','and','to','of','in','is','that','it','for','on','with','as','this',
            'are','be','was','by','an','at','from','or','not','have','has','had'
        }
        return id_fallback | en_fallback

# HALAMAN BERANDA
if page == "home":
    st.markdown(
        "<h2 style='text-align:center'>Selamat datang di <b>UlasAnalisa</b></h2>",
        unsafe_allow_html=True,
    )

    st.markdown("### Apa itu **UlasAnalisa?**")
    st.markdown(
        """
        **UlasAnalisa** adalah website untuk menganalisis sentimen ulasan aplikasi di Google Play Store secara otomatis
        dan menyajikannya dalam tabel yang mudah dipahami.  
        Hasil sentimen juga dapat diunduh dalam format **.csv**.
        """
    )

    st.divider()
    st.markdown("### Bagaimana Cara Memakainya?")

    st.markdown("#### Jika membukanya menggunakan **desktop**")

    # Step 1 (website)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image("static/1.png", caption="Tampilan Google Play (Website)", use_container_width=True)
    with c2:
        st.markdown("### Step 1 (Website)")
        st.write("Copy tautan aplikasi dari halaman Google Play Store yang ingin dianalisis.")

    # Step 2 
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image("static/5.png", use_container_width=True)
    with c2:
        st.markdown("### Step 2")
        st.write("Buka halaman **Prediksi** di UlasAnalisa, lalu **paste** tautan aplikasi pada kolom yang disediakan.")

    # Step 3
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image("static/6.png", use_container_width=True)
    with c2:
        st.markdown("### Step 3")
        st.write("Atur **model**, **bahasa**, **negara**, **jumlah ulasan**, dan **urutan** sesuai kebutuhan.")

    # Step 4 
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image("static/7.png", use_container_width=True)
    with c2:
        st.markdown("### Step 4")
        st.write("Klik **Prediksi** untuk mulai menganalisis. Hasil dan tombol **Download .csv** akan muncul.")

    st.divider()
    
    st.markdown("#### Jika membukanya menggunakan **mobile**")
    MOBILE_W = 180  

    def center_image(path, width=MOBILE_W):
        L, C, R = st.columns([1, 2, 1])   
        with C:
            st.image(path, width=width)

    # Step 1 (mobile)
    st.markdown("### Step 1")
    st.write("Buka Google Play Store, pilih aplikasinya, ketuk **⋮ → Share → Copy URL**.")
    c1, c2, c3 = st.columns(3)
    with c1: st.image("static/2.png", width=MOBILE_W)
    with c2: st.image("static/3.png", width=MOBILE_W)
    with c3: st.image("static/4.png", width=MOBILE_W)
    
    # Step 2
    st.markdown("### Step 2")
    st.write("Masuk ke website **UlasAnalisa** lalu ke halaman **Prediksi**, kemudian **paste** tautan aplikasi pada kolom yang disediakan.")
    center_image("static/8.png")

    # Step 3
    st.markdown("### Step 3")
    st.write("Ketuk tombol **Prediksi** untuk mulai.")
    center_image("static/9.png")

    # Step 4
    st.markdown("### Step 4")
    st.write("Sesuaikan **Pengaturan** (model/bahasa/negara/jumlah ulasan/urutan).")
    center_image("static/10.png")

    # Step 5
    st.markdown("### Step 5")
    st.write("Gunakan ikon/tombol untuk membuka panel **Pengaturan** bila diperlukan.")
    center_image("static/11.png")

# HALAMAN TENTANG
# ================== HALAMAN TENTANG (final, kecil & rapi) ==================
elif page == "tentang":
    IMG_W  = 220   # ukuran foto kiri/kanan (samakan)
    LOGO_W = 300   # ukuran logo UNTAR

    st.markdown("""
    <style>
    /* 1) Kecilkan padding default konten Streamlit di bawah navbar */
    .block-container { padding-top: 0.6rem; }   
    /* 2) Tarik judul ke atas dengan margin-top negatif */
    .about-title { margin: -90px 0 14px; font-weight: 800; } 
    /* Versi desktop – boleh beda dari mobile */
    @media (min-width: 992px){
    .about-title { margin: -110px 0 16px; }
    }

    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .about-title-wrap{
    width:100%;
    display:flex;
    justify-content:center;   /* center horizontal */
    align-items:center;       /* center vertical (tinggi elemen) */
    margin: 6px 0 22px;       /* atur jarak atas/bawah */
    }
    .about-title-wrap span{
    font-weight:800;
    font-size: clamp(26px, 3.0vw, 40px);
    line-height:1.15;
    text-align:center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Judul
    st.markdown(
        "<div class='about-title-wrap'><span>Pengembang Website</span></div>",
        unsafe_allow_html=True
    )

    # --- baris foto bersebelahan & tetap di tengah ---
    spacerL, left, right, spacerR = st.columns([1.60, 3, 3, 0.65], gap="small")

    # >>> FOTO KIRI (geser sedikit ke kanan pakai sub-kolom off/area; posisi gambar TIDAK DIUBAH)
    with left:
        off, area = st.columns([0.35, 0.65])
        with area:
            # wrapper selebar foto -> caption pas di tengah bawah foto (bukan tengah kolom)
            st.markdown(f"<div style='width:{IMG_W}px;'>", unsafe_allow_html=True)
            st.image("static/fotoku.jpg", width=IMG_W)
            st.markdown(
                f"<div class='cap' style='width:{IMG_W}px;'>"
                "<b>Nama :</b> Parveen Uzma Habidin<br>"
                "<b>NIM :</b> 535220226"
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # >>> FOTO KANAN
    with right:
        st.markdown(f"<div style='width:{IMG_W}px;'>", unsafe_allow_html=True)
        st.image("static/pak_tri.jpg", width=IMG_W)
        st.markdown(
            f"<div class='cap' style='width:{IMG_W}px;'>"
            "<b>Dosen Pembimbing</b><br>"
            "Tri Sutrisno, S.Si., M.Sc."
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")  # spasi kecil

    # --- logo UNTAR tepat di tengah; caption dibatasi lebar logo agar pas di bawah logo ---
    a, b, c = st.columns([1.3, 1, 0.7])
    with b:
        st.markdown(f"<div style='width:{LOGO_W}px; margin:0 auto;'>", unsafe_allow_html=True)
        st.image("static/Logo_untar.png", width=LOGO_W)
        
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# HALAMAN PREDIKSI
elif page == "prediksi":

    st.title("Prediksi Sentimen dari Link Google Play")
    st.caption("Masukkan link aplikasi dari Google Play Store, lalu sistem akan prediksi sentimennya")

    #ARTIFACT PATH
    VEC_PATH = Path("Artifacts") / "tfidf_vectorizer.joblib"
    SVM_PATH = Path("Artifacts") / "svm_rbf_model.joblib"
    RF_PATH  = Path("Artifacts") / "random_forest_model.joblib"

    @st.cache_resource
    def load_artifacts():
        vec = joblib.load(VEC_PATH)
        svm = joblib.load(SVM_PATH) if SVM_PATH.exists() else None
        rf  = joblib.load(RF_PATH)  if RF_PATH.exists()  else None
        return vec, svm, rf

    try:
        tfidf_vectorizer, svm_model, rf_model = load_artifacts()
    except Exception as e:
        st.error(f"Gagal memuat artifacts.\nDetail: {e}")
        st.stop()

    #MODEL
    avail = []
    if svm_model is not None: avail.append("SVM (RBF)")
    if rf_model  is not None: avail.append("RandomForest")
    if svm_model is not None and rf_model is not None: avail.append("SVM dan RandomForest")
    if not avail:
        st.error("Tidak ada model yang tersedia.")
        st.stop()

    #SIDEBAR
    with st.sidebar:
        st.header("Pengaturan")
        model_name = st.selectbox("Pilih model", avail, index=0)
        lang       = st.selectbox("Bahasa ulasan", ["id", "en"], index=0)
        country    = st.selectbox("Negara", ["id", "us"], index=0)
        n_reviews  = st.slider("Jumlah ulasan di-scrape", 50, 1000, 200, 50)
        sort_opt   = st.selectbox("Urutkan", ["NEWEST", "MOST_RELEVANT"], index=0)
        run        = st.button("Prediksi")

    #HELPER SCRAPE
    ID_RE = re.compile(r"[?&]id=([a-zA-Z0-9._]+)")
    def parse_app_id(text: str) -> str:
        t = (text or "").strip()
        m = ID_RE.search(t)
        return m.group(1) if m else t

    def scrape_reviews(app_id: str, lang="id", country="id", n=200, sort="NEWEST"):
        sort_map = {"NEWEST": Sort.NEWEST, "MOST_RELEVANT": Sort.MOST_RELEVANT}
        sort = sort_map.get(sort, Sort.NEWEST)
        got, token = [], None
        while len(got) < n:
            batch, token = reviews(
                app_id, lang=lang, country=country, sort=sort,
                count=min(200, n-len(got)), continuation_token=token
            )
            got.extend(batch)
            if token is None:
                break
        if not got:
            return pd.DataFrame(columns=["content", "score", "at", "replyContent", "userName"])
        return pd.DataFrame(got)
    
    st.session_state["scrape_n"] = n_reviews
    
    #helper simpan figure ke bytes (PNG)
    def fig_to_png_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
        buf.seek(0)
        return buf.read()

    #INPUT LINK
    link = st.text_input(
        "Masukkan link Google Play / package id",
        placeholder="https://play.google.com/store/apps/details?id=com.zhiliaoapp.musically"
    )

    #PROSES PREDIKSI
    if run:
        app_id = parse_app_id(link)
        if not app_id:
            st.error("Package id tidak valid.")
            st.stop()

        try:
            meta = gp_app(app_id, lang=lang, country=country)
            st.markdown(
                f"**App:** {meta.get('title','?')}  \n"
                f"**Package:** `{app_id}`  \n"
                f"**Installs:** {meta.get('installs','?')}  \n"
                f"**Score:** {meta.get('score','?')}"
            )
        except Exception:
            st.info(f"Package: `{app_id}`")
            
        #ambil ulasan
        with st.spinner(f"Mengambil {n_reviews} ulasan..."):
            df = scrape_reviews(app_id, lang=lang, country=country, n=n_reviews, sort=sort_opt)
        if df.empty:
            st.warning("Tidak ada ulasan yang diambil.")
            st.stop()

        #merapikan kolom
        df = df.rename(columns={"content": "text", "score": "rating", "at": "date"})
        cols = ["text", "rating", "date", "userName", "replyContent"]
        df  = df[[c for c in cols if c in df.columns]].copy()

        #tf-idf + prediksi
        with st.spinner("Mengubah fitur (TF-IDF) dan memprediksi"):
            X = tfidf_vectorizer.transform(df["text"].astype(str)).toarray()
            results = {}
            if model_name == "SVM (RBF)":
                y = svm_model.predict(X)
                t = df.copy(); t["pred"] = y; t["pred_label"] = t["pred"].map({1:"Positive", 0:"Negative"})
                results["SVM (RBF)"] = t
            elif model_name == "RandomForest":
                y = rf_model.predict(X)
                t = df.copy(); t["pred"] = y; t["pred_label"] = t["pred"].map({1:"Positive", 0:"Negative"})
                results["RandomForest"] = t
            else:
                y_svm = svm_model.predict(X)
                t1 = df.copy(); t1["pred"] = y_svm; t1["pred_label"] = t1["pred"].map({1:"Positive", 0:"Negative"})
                results["SVM (RBF)"] = t1
                y_rf  = rf_model.predict(X)
                t2 = df.copy(); t2["pred"] = y_rf; t2["pred_label"] = t2["pred"].map({1:"Positive", 0:"Negative"})
                results["RandomForest"] = t2

        st.session_state.results = results
        st.session_state.app_id  = app_id
        st.session_state.is_combo = (model_name == "SVM dan RandomForest")

        # 2 model
        if st.session_state.is_combo:
            df_svm = results["SVM (RBF)"]
            df_rf  = results["RandomForest"]

            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as w:
                df_svm.to_excel(w, sheet_name="SVM (RBF)", index=False)
                df_rf.to_excel(w,  sheet_name="RandomForest", index=False)
            out.seek(0)
            st.session_state.csv_pred = out.getvalue()

            dist_svm = df_svm["pred_label"].value_counts().rename_axis("sentiment").reset_index(name="count")
            dist_rf  = df_rf["pred_label"].value_counts().rename_axis("sentiment").reset_index(name="count")
            out2 = io.BytesIO()
            with pd.ExcelWriter(out2, engine="xlsxwriter") as w:
                dist_svm.to_excel(w, sheet_name="SVM (RBF)", index=False)
                dist_rf.to_excel(w,  sheet_name="RandomForest", index=False)
            out2.seek(0)
            st.session_state.csv_dist = out2.getvalue()
        else:
            key = next(iter(results))
            dfk = results[key]
            st.session_state.csv_pred = dfk.to_csv(index=False).encode("utf-8")
            st.session_state.csv_dist = (
                dfk["pred_label"]
                .value_counts()
                .rename_axis("sentiment")
                .reset_index(name="count")
                .to_csv(index=False)
                .encode("utf-8")
            )

#Helper batas baris 
def table_limit(df: pd.DataFrame) -> int:
    n = st.session_state.get("scrape_n", 0)
    return min(len(df), n + 500)

# OUTPUT / HASIL
if st.session_state.results and page == "prediksi":

    all_fig_images = [] 

    #Distribusi Hasil Sentimen
    #BAR
    items = list(st.session_state.results.items())
    cols = st.columns(len(items))
    for c, (name, dfm) in zip(cols, items):
        with c:
            st.subheader(f"Distribusi Sentimen – {name}")

            dist = (
                dfm["pred_label"].value_counts()
                .reindex(["Positive","Negative"])
                .fillna(0).astype(int)
                .rename_axis("Sentiment").reset_index(name="Count")
            )
            dist["Legend"] = dist.apply(lambda r: f"{r['Sentiment']} ({r['Count']})", axis=1)

            LEGEND_POS = alt.Legend(
            title=None,
            orient="top",          
            direction="horizontal",   
            labelLimit=1000,
            symbolSize=120,
            padding=8
        )
            CHART_W, BAR_H = 560, 340 
            bar = alt.Chart(dist).mark_bar().encode(
                x=alt.X("Sentiment:N", sort=["Positive", "Negative"], title="Sentiment"),
                y=alt.Y("Count:Q", title="Count",
                        axis=alt.Axis(labelColor="white", titleColor="white", labelPadding=4)),
                color=alt.Color("Legend:N", legend=LEGEND_POS),
                tooltip=[alt.Tooltip("Sentiment:N"), alt.Tooltip("Count:Q")]
            )
            
            labels = alt.Chart(dist).mark_text(dy=-6).encode(
                x="Sentiment:N", y="Count:Q", text="Count:Q", color=alt.value("black")
            )
            
            st.altair_chart(bar + labels, use_container_width=True)

            #PIE
            PIE_W, PIE_H = 520, 420          
            OUTER_R = 190                    
            INNER_R = 20
            LABEL_R = (OUTER_R + INNER_R) // 2
 
            LEGEND_POS = alt.Legend(
                title=None,
                orient="bottom",          
                direction="horizontal",
                labelLimit=160,
                symbolSize=120,
                padding=8
            )

            base = (
                alt.Chart(dist)
                .transform_joinaggregate(total='sum(Count)')
                .transform_calculate(
                    pct='datum.Count / datum.total',
                    pct_txt="format(datum.Count / datum.total, '.1%')"
                )
            )

            arc = base.mark_arc(innerRadius=70).encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color("Legend:N", legend=LEGEND_POS),
                tooltip=[
                    alt.Tooltip("Legend:N", title="Sentiment"),
                    alt.Tooltip("Count:Q",  title="Jumlah"),
                    alt.Tooltip("pct_txt:N", title="Proporsi")
                ]
            )

            LABEL_OFFSET = 10  

            labels_outline = (
                base.transform_filter(alt.datum.pct >= 0.03)
                .mark_text(
                    radius=LABEL_R + LABEL_OFFSET,
                    dx=2, dy=-1,
                    fontWeight="bold",
                    fontSize=14,          
                    color="black",        
                )
                .encode(
                    theta=alt.Theta("Count:Q", stack=True),
                    text=alt.Text("pct_txt:N"),
                )
            )

            labels_fill = (
                base.transform_filter(alt.datum.pct >= 0.03)
                .mark_text(
                    radius=LABEL_R + LABEL_OFFSET,
                    dx=2, dy=-1,
                    fontWeight="bold",
                    fontSize=12,          
                    color="white",        
                )
                .encode(
                    theta=alt.Theta("Count:Q", stack=True),
                    text=alt.Text("pct_txt:N"),
                )
            )

            labels = labels_outline + labels_fill    

            pie = (arc + labels).properties(
                title=f"Proporsi Sentimen – {name}",
                width=PIE_W, height=PIE_H, padding={"left":10,"right":10,"top":10,"bottom":10}
            )

            st.altair_chart(pie, use_container_width=True)

            # sheet grafik
            fig_bar, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
            ax.bar(dist["Sentiment"], dist["Count"])
            ax.set_xlabel("Sentiment"); ax.set_ylabel("Count"); ax.set_title(f"Distribusi – {name}")
            all_fig_images.append((f"{name}_distribusi_bar.png", fig_to_png_bytes(fig_bar)))

            fig_pie, ax = plt.subplots(figsize=(7.2, 5.0), dpi=120)
            ax.pie(dist["Count"], labels=dist["Sentiment"], autopct="%.1f%%", startangle=120)
            ax.set_title(f"Proporsi Sentimen – {name}"); ax.axis("equal")
            all_fig_images.append((f"{name}_distribusi_pie.png", fig_to_png_bytes(fig_pie)))

            # tabel sampel
            st.subheader(f"Sampel Hasil Prediksi – {name}")
            st.dataframe(dfm.head(table_limit(dfm)), use_container_width=True)

    #WordCloud
    st.subheader("Visualisasi WordCloud")
    st.caption(
        "WordCloud menampilkan kata/frasa yang paling sering muncul pada kumpulan teks. "
        "Semakin besar ukuran huruf, semakin sering kata tersebut muncul. "
        "Warna hanya tema visual dan bukan penanda sentimen. Pemisahan sentimen dilakukan "
        "dengan menampilkan WordCloud terpisah untuk Positive dan Negative."
    )
    
    for model_name, dfm in st.session_state.results.items():
        st.markdown(f"### {model_name}")

        c_pos, c_neg = st.columns(2)

        for label, col in [("Positive", c_pos), ("Negative", c_neg)]:
            sub = dfm.loc[dfm["pred_label"] == label, "text"].dropna()
            if sub.empty: 
                continue

            text = " ".join(sub.astype(str))
            stop_words = ensure_stopwords()

            wc = WordCloud(
                width=420, height=230, background_color="white",
                stopwords=stop_words, colormap="coolwarm",
                max_words=100, prefer_horizontal=0.9
            ).generate(text)

            fig_wc, ax = plt.subplots(figsize=(3.6, 2.1), dpi=100)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{label}", fontsize=12)
            with col:
                st.pyplot(fig_wc, use_container_width=False)
            all_fig_images.append((f"{model_name}_wordcloud_{label}.png", fig_to_png_bytes(fig_wc))) 
            
            # Ambil 5 kata terbanyak dari token yang sama dipakai untuk barplot
            tokens = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
            tokens = [t for t in tokens if t not in stop_words and len(t) >= 3]
            top5 = [w for w, c in Counter(tokens).most_common(5)]
            if top5:
                st.markdown(
                    f"**Penjelasannya:** Untuk sentimen **{label}** pada **{model_name}**, "
                    f"kata/fras­a yang dominan antara lain: `{', '.join(top5)}`. "
                    "Ukuran huruf yang besar mengindikasikan frekuensi tinggi sehingga tema di atas "
                    "menjadi topik yang paling sering dibahas dalam kelompok sentimen ini."
                ) 

    #Confusion Matriks
    st.subheader("Confusion Matrix per Model")

    def plot_cm_v2(cm: np.ndarray, title: str):
        fig, ax = plt.subplots(figsize=(3.9, 3.2), dpi=110)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(int(cm.max()), 1))
        ax.set_aspect("equal")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred POS", "Pred NEG"], fontsize=10)
        ax.set_yticklabels(["True POS", "True NEG"], fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=10, labelpad=2)
        ax.set_ylabel("True label", fontsize=10, labelpad=2)
        ax.set_title(f"Confusion Matrix — {title}", fontsize=12, pad=6)

        thr = cm.max()/2 if cm.max() > 0 else 0.5
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{int(cm[i, j])}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black",
                    fontsize=11, fontweight="bold"
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.ax.tick_params(labelsize=9)
        fig.tight_layout(pad=0.2)
        return fig

    # Bangun CM per model dari st.session_state.results
    pairs = []
    for model_name, dfm in st.session_state.results.items():
        tmp = dfm.copy()

        # true label dari rating bintang: >=4 -> 1 (POS), <4 -> 0 (NEG)
        tmp["true_label"] = tmp["rating"].apply(
            lambda r: 1 if pd.notna(r) and r >= 4 else (0 if pd.notna(r) else np.nan)
        )
        tmp = tmp.dropna(subset=["true_label"])
        if tmp.empty:
            continue

        y_true = tmp["true_label"].astype(int).values
        y_pred = tmp["pred"].astype(int).values   # asumsi 1=POS, 0=NEG
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        pairs.append((model_name, cm))

    if not pairs:
        st.info("Tidak ada data untuk membuat Confusion Matrix.")
    else:
        cols = st.columns(2, gap="small")
        for i, (name, cm) in enumerate(pairs):
            with cols[i % 2]:
                fig_cm = plot_cm_v2(cm, name)
                st.pyplot(fig_cm, use_container_width=True)

                # Penjelasan & metrik cepat dari CM
                TP, FN = int(cm[0, 0]), int(cm[0, 1])  # baris True POS
                FP, TN = int(cm[1, 0]), int(cm[1, 1])  # baris True NEG
                total = TP + TN + FP + FN or 1
                acc  = (TP + TN) / total
                prec = TP / (TP + FP) if (TP + FP) else 0.0
                rec  = TP / (TP + FN) if (TP + FN) else 0.0
                f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

                st.markdown(
                    f"""
                **Keterangan:**  
                - **Baris** = label **asli** (kebenaran data), **Kolom** = label **prediksi** (hasil model).

                - **TP = {TP}** → Asli **positif**, diprediksi **positif** → **tepat**.  
                - **FN = {FN}** → Asli **positif**, diprediksi **negatif** → **terlewat** (positif tidak tertangkap).  
                - **FP = {FP}** → Asli **negatif**, diprediksi **positif** → **alarm palsu** (negatif dikira positif).  
                - **TN = {TN}** → Asli **negatif**, diprediksi **negatif** → **tepat**.

                > Ingat: **FP** = salah “menganggap positif”, **FN** = salah “melewatkan positif”.
                """
                )

                # Opsional: simpan gambar jika kamu koleksi untuk unduhan
                if 'all_fig_images' in globals():
                    all_fig_images.append(
                        (f"{name}_confusion_matrix.png", fig_to_png_bytes(fig_cm))
                    )


    #Metrik Evaluasi
    def rating_to_label(r):
        if pd.isna(r): return None
        return 1 if r >= 4 else 0

    metrics = []
    for name, dfm in st.session_state.results.items():
        d = dfm.copy()
        d["true_label"] = d["rating"].apply(rating_to_label)
        d = d.dropna(subset=["true_label"])
        if d.empty: 
            continue
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(d["true_label"], d["pred"]),
            "Precision": precision_score(d["true_label"], d["pred"]),
            "Recall": recall_score(d["true_label"], d["pred"]),     # ← ganti ini
            "F1-Score": f1_score(d["true_label"], d["pred"]),
        })

    if metrics:
        st.subheader("Perbandingan Metrik Evaluasi")
        df_metrics = pd.DataFrame(metrics)
        st.dataframe(df_metrics, use_container_width=True)
        
        # ---- Penjelasan tabel metrik ----
        # Urutkan: prioritas F1-Score, lalu Accuracy sebagai tie-breaker
        ranked = df_metrics.sort_values(
            by=["F1-Score", "Accuracy"], ascending=[False, False]
        ).reset_index(drop=True)

        best = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else None

        # Model terbaik menurut metrik lain (berguna untuk catatan)
        best_acc  = df_metrics.loc[df_metrics["Accuracy"].idxmax()]
        best_prec = df_metrics.loc[df_metrics["Precision"].idxmax()]
        best_rec  = df_metrics.loc[df_metrics["Recall"].idxmax()]

        # Narasi trade-off: lihat keseimbangan precision vs recall
        if float(best["Precision"]) > float(best["Recall"]):
            tradeoff = (
                "Model ini **lebih ketat** saat menyatakan *positif* "
                "(**Precision** lebih tinggi → **false positive** lebih sedikit), "
                "namun ada potensi **positif terlewat** lebih banyak dibanding model dengan Recall lebih tinggi."
            )
        elif float(best["Precision"]) < float(best["Recall"]):
            tradeoff = (
                "Model ini **lebih menyeluruh** menangkap *positif* "
                "(**Recall** lebih tinggi → **false negative** lebih sedikit), "
                "namun bisa sedikit lebih banyak memberikan **alarm palsu** dibanding model dengan Precision lebih tinggi."
            )
        else:
            tradeoff = (
                "Model ini memiliki keseimbangan yang sangat baik antara **Precision** dan **Recall**."
            )

        # (Opsional) bedakan jika ada seri F1-Score
        tie_note = ""
        tied = ranked[ranked["F1-Score"].eq(best["F1-Score"])]
        if len(tied) > 1:
            # jika seri, pilih yang accuracy tertinggi; jelaskan
            alternatives = ", ".join(tied["Model"].tolist()[1:])
            tie_note = (
                f"\n\n> Catatan: Ada **seri F1-Score** antara **{best['Model']}** dan {alternatives}. "
                "Pemilihan dilakukan dengan *tie-breaker* **Accuracy** yang lebih tinggi."
            )

        # Teks ringkasan yang mudah dipahami
        st.markdown(
            f"""
        **Penjelasan**  
        Model terbaik menurut **F1-Score** adalah **{best['Model']}**  
        dengan **F1 ≈ {best['F1-Score']:.3f}**, **Accuracy ≈ {best['Accuracy']:.3f}**,  
        **Precision ≈ {best['Precision']:.3f}**, dan **Recall ≈ {best['Recall']:.3f}**.

        {tradeoff}

        """
            + (
                f"Model peringkat berikutnya: **{second['Model']}** "
                f"(F1 ≈ {second['F1-Score']:.3f}, Accuracy ≈ {second['Accuracy']:.3f}).\n\n"
                if second is not None else ""
            )
            + f"""
        
        **Rekomendasi pemilihan model:**
        - Pilih **{best['Model']}** bila kamu ingin **keseimbangan** terbaik (F1 tinggi) — aman saat data tidak seimbang.
        - Jika *false positive* harus **sangat** rendah (hindari “positif palsu”), pertimbangkan model dengan **Precision** tertinggi.
        - Jika *false negative* harus **sangat** rendah (jangan sampai ada yang terlewat), pertimbangkan model dengan **Recall** tertinggi.
        """
            + tie_note
        )
    else:
        st.info("Tidak ada metrik yang bisa dihitung.")


    #TOMBOL DOWNLOAD
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:

        # tulis sheet per model
        for i, (name, dfm) in enumerate(st.session_state.results.items(), start=1):
            sheet_name = name if len(name) <= 31 else f"Model{i}"
            dfm.head(table_limit(dfm)).to_excel(w, sheet_name=sheet_name, index=False) 

        # sheet grafik
        ws_name = "Grafik"
        w.book.add_worksheet(ws_name)
        ws = w.sheets[ws_name]

        # info judul
        ws.write(0, 0, "Semua Grafik")
        row, col = 2, 0
        for fname, png in all_fig_images:
            ws.write(row, col, fname)
            ws.insert_image(row+1, col, fname, {"image_data": io.BytesIO(png), "x_scale": 0.9, "y_scale": 0.9})
            row += 22  

    out.seek(0)
    combined_xlsx = out.getvalue()

    st.markdown("---")
    st.download_button(
        "Download Hasil Prediksi",
        data=combined_xlsx,
        file_name=f"{st.session_state.app_id}_hasil_prediksi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        key="dl_all_in_one"
    )

elif page == "prediksi" and not st.session_state.results:
    st.info("Masukkan link/package, lalu klik **Prediksi**.")
