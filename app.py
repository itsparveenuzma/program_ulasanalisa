import re
import base64
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
from google_play_scraper import app as gp_app, reviews, Sort

# PAGE CONFIG
st.set_page_config(
    page_title="UlasAnalisa – Prediksi Sentimen",
    page_icon="static/logo_ulas.png", 
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# HALAMAN BERANDA
if page == "home":
    st.markdown("## Selamat datang di **UlasAnalisa**")
    st.markdown("### Apa itu **UlasAnalisa?**")
    st.markdown(
        """
        **UlasAnalisa** adalah website yang membantu menganalisis sentimen ulasan aplikasi di Google Play Store secara otomatis dan menyajikannya dalam bentuk tabel yang mudah dipahami.  
        Hasil sentimen bisa diunduh dalam bentuk **.csv**.
        """
    )

    st.markdown("### Bagaimana Cara Memakainya?")

    # STEP 1 (Website)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("static/1.png", caption="Tampilan Google Play di Website", use_container_width=True)
    with col2:
        st.markdown("### Step 1 (Website)")
        st.write("Copy link aplikasi dari halaman Google Play Store yang ingin dianalisa (website).")

    st.markdown("---")

    # STEP 1 (Handphone)
    st.markdown("### Step 1 (Handphone)")
    sp1, c1, c2, c3, sp2 = st.columns([1, 2, 2, 2, 1])
    with c1:
        st.image("static/2.png", width=230)

    with c2:
        st.image("static/3.png", width=230)

    with c3:
        st.image("static/4.png", width=230)
    st.write(
        "Buka Google Play Store di HP → cari aplikasinya → ketuk **⋮ → Share** → pilih **Copy URL**."
    )

    st.markdown("---")

    # STEP 2
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("static/5.png", use_container_width=True)
    with col2:
        st.markdown("### Step 2")
        st.write("Paste / tempel link URL tadi ke kolom input di halaman **Prediksi**.")

    st.markdown("---")

    # STEP 3
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("static/6.png", use_container_width=True)
    with col2:
        st.markdown("### Step 3")
        st.write("Atur pengaturan (model, bahasa, negara, jumlah ulasan, urutan) sesuai kebutuhan.")

    st.markdown("---")

    # STEP 4
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("static/7.png", use_container_width=True)
    with col2:
        st.markdown("### Step 4")
        st.write("Klik tombol **Prediksi** → sistem akan ambil ulasan dan menampilkan hasil serta tombol download CSV.")

# HALAMAN TENTANG
elif page == "tentang":
    st.markdown("### Pengembang Website")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("static/fotoku.png", width=180)
    with col2:
        st.markdown(
            """
            **Nama** : Parveen Uzma Habidin  
            **NIM** : 535220226  
            **Jurusan** : Teknik Informatika  
            **Fakultas** : Teknik Informasi  

            **Topik Skripsi** :  
            Perencanaan Analisis Sentimen Aplikasi Sosial Media Pada Google Play Store Menggunakan Model Random Forest, Support Vector Machine dan TF-IDF
            """
        )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dosen Pembimbing")
        st.image("static/pak_tri.webp", width=140)
        st.markdown("Tri Sutrisno, S.Si., M.Sc.")
    with col2:
        st.markdown("### Institusi")
        st.image("static/logo_untar.png", width=180)
        st.markdown("**Universitas Tarumanagara**")


#HALAMAN PREDIKSI
elif page == "prediksi":
    st.title("Prediksi Sentimen dari Link Google Play")
    st.caption("Masukkan link aplikasi dari Google Play Store, lalu sistem akan prediksi sentimennya")

    #ARTIFACTS PATH
    VEC_PATH = Path("Artifacts") / "tfidf_vectorizer.joblib"
    SVM_PATH = Path("Artifacts") / "svm_rbf_model.joblib"
    RF_PATH = Path("Artifacts") / "random_forest_model.joblib"

    @st.cache_resource
    def load_artifacts():
        vec = joblib.load(VEC_PATH)
        svm = joblib.load(SVM_PATH) if SVM_PATH.exists() else None
        rf = joblib.load(RF_PATH) if RF_PATH.exists() else None
        return vec, svm, rf

    try:
        tfidf_vectorizer, svm_model, rf_model = load_artifacts()
    except Exception as e:
        st.error(
            f"Gagal memuat artifacts.\n"
            f"Vectorizer: `{VEC_PATH}` (wajib)\n"
            f"Model: `{SVM_PATH}` atau `{RF_PATH}` (minimal salah satu)\n\n"
            f"Detail: {e}"
        )
        st.stop()

    #SESSION STATE
    for k, v in dict(pred_df=None, dist_df=None, app_id=None, csv_pred=None, csv_dist=None).items():
        if k not in st.session_state:
            st.session_state[k] = v

    #MODEL
    avail = []
    if svm_model is not None:
        avail.append("SVM (RBF)")
    if rf_model is not None:
        avail.append("RandomForest")
    if not avail:
        st.error(
            "Tidak ada model yang tersedia. Letakkan minimal salah satu model: "
            "SVM (`svm_rbf_model.joblib`) atau RF (`random_forest_model.joblib`)."
        )
        st.stop()

    #SIDEBAR
    st.markdown("""
    <style>
    /* geser sidebar biar ga ketutup navbar */
    [data-testid="stSidebar"] {
        top: 90px;
    }

    /* geser tombol collapse-nya juga */
    [data-testid="stSidebarCollapseButton"] {
        top: 95px;
        left: 6px;
        z-index: 100000;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Pengaturan")
        model_name = st.selectbox("Pilih model", avail, index=0)
        lang = st.selectbox("Bahasa ulasan", ["id", "en"], index=0)
        country = st.selectbox("Negara", ["id", "us"], index=0)
        n_reviews = st.slider("Jumlah ulasan di-scrape", 50, 1000, 200, 50)
        sort_opt = st.selectbox("Urutkan", ["NEWEST", "MOST_RELEVANT"], index=0)
        run = st.button("Prediksi")

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
                app_id,
                lang=lang,
                country=country,
                sort=sort,
                count=min(200, n - len(got)),
                continuation_token=token,
            )
            got.extend(batch)
            if token is None:
                break
        if not got:
            return pd.DataFrame(
                columns=["content", "score", "at", "replyContent", "userName"]
            )
        return pd.DataFrame(got)

    #INPUT LINK
    link = st.text_input(
        "Masukkan link Google Play / package id",
        placeholder="https://play.google.com/store/apps/details?id=com.zhiliaoapp.musically",
    )

    #PROSES PREDIKSI
    if run:
        app_id = parse_app_id(link)
        if not app_id:
            st.error("Package id tidak valid.")
            st.stop()

        # meta app
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

        # ambil ulasan
        with st.spinner(f"Mengambil {n_reviews} ulasan..."):
            df = scrape_reviews(
                app_id, lang=lang, country=country, n=n_reviews, sort=sort_opt
            )
        if df.empty:
            st.warning("Tidak ada ulasan yang diambil.")
            st.stop()

        # rapikan kolom
        df = df.rename(columns={"content": "text", "score": "rating", "at": "date"})
        cols = ["text", "rating", "date", "userName", "replyContent"]
        df = df[[c for c in cols if c in df.columns]].copy()

        # tfidf + prediksi
        with st.spinner("Mengubah fitur (TF-IDF) dan memprediksi"):
            X_tfidf = tfidf_vectorizer.transform(df["text"].astype(str))
            X_dense = X_tfidf.toarray()
            model = svm_model if model_name == "SVM (RBF)" else rf_model
            if model is None:
                st.error(f"Model {model_name} tidak tersedia.")
                st.stop()
            y_pred = model.predict(X_dense)

        sent_map = {1: "Positive", 0: "Negative"}
        df["pred"] = y_pred
        df["pred_label"] = df["pred"].map(sent_map)
        dist = (
            df["pred_label"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )

        st.session_state.app_id = app_id
        st.session_state.pred_df = df
        st.session_state.dist_df = dist
        st.session_state.csv_pred = df.to_csv(index=False).encode("utf-8")
        st.session_state.csv_dist = dist.to_csv(index=False).encode("utf-8")

    #TAMPILKAN HASIL
    if st.session_state.pred_df is not None:
        st.subheader("Distribusi Sentimen (Prediksi)")
        dist_df = st.session_state.dist_df.set_index("Sentiment")
        st.bar_chart(dist_df)

        st.subheader("Sampel Hasil Prediksi")
        st.dataframe(st.session_state.pred_df.head(20), use_container_width=True)

        left, middle, right = st.columns(3)
        with left:
            st.download_button(
                "Download Hasil Prediksi (CSV)",
                data=st.session_state.csv_pred,
                file_name=f"{st.session_state.app_id}_prediksi_ulasan.csv",
                mime="text/csv",
                key="dl_pred",
                type="primary",
            )
        with middle:
            st.download_button(
                "Download Distribusi Sentimen (CSV)",
                data=st.session_state.csv_dist,
                file_name=f"{st.session_state.app_id}_distribusi_sentimen.csv",
                mime="text/csv",
                key="dl_dist",
                type="primary",
            )
    else:
        st.info("Masukkan link/package, lalu klik **Prediksi**.")
