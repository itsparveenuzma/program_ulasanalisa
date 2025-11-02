import re
import io
import base64
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
from google_play_scraper import app as gp_app, reviews, Sort
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    st.divider()

    # Step 2 
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image("static/5.png", use_container_width=True)
    with c2:
        st.markdown("### Step 2")
        st.write("Buka halaman **Prediksi** di UlasAnalisa, lalu **paste** tautan aplikasi pada kolom yang disediakan.")

    st.divider()

    # Step 3
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image("static/6.png", use_container_width=True)
    with c2:
        st.markdown("### Step 3")
        st.write("Atur **model**, **bahasa**, **negara**, **jumlah ulasan**, dan **urutan** sesuai kebutuhan.")

    st.divider()

    # Step 4 
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image("static/7.png", use_container_width=True)
    with c2:
        st.markdown("### Step 4")
        st.write("Klik **Prediksi** untuk mulai menganalisis. Hasil dan tombol **Download .csv** akan muncul.")

    st.divider()

    st.markdown("#### Jika membukanya menggunakan **mobile**")

    # Step 1 (mobile)
    g1, g2, g3 = st.columns([1, 1, 1])
    with g1: st.image("static/2.png", use_container_width=True)
    with g2: st.image("static/3.png", use_container_width=True)
    with g3: st.image("static/4.png", use_container_width=True)
    st.markdown("**Step 1 (Mobile)** – Buka Google Play Store, pilih aplikasinya, ketuk **⋮ → Share → Copy URL**.")

    st.divider()

    # Step 2 
    m1, m2 = st.columns([1, 1])
    with m1: st.image("static/8.png", use_container_width=True)
    with m2:
        st.markdown("### Step 2 (Mobile)")
        st.write("Masuk ke website **UlasAnalisa** → buka halaman **Prediksi** → **paste** tautan aplikasi.")

    st.divider()

    # Step 3 (
    m1, m2 = st.columns([1, 1])
    with m1: st.image("static/9.png", use_container_width=True)
    with m2:
        st.markdown("### Step 3 (Mobile)")
        st.write("Ketuk tombol **Prediksi** untuk mulai.")

    st.divider()

    # Step 4 
    m1, m2 = st.columns([1, 1])
    with m1: st.image("static/10.png", use_container_width=True)
    with m2:
        st.markdown("### Step 4 (Mobile)")
        st.write("Sesuaikan **Pengaturan** (model/bahasa/negara/jumlah ulasan/urutan) sesuai kebutuhan.")

    st.divider()

    # Step 5 
    m1, m2 = st.columns([1, 1])
    with m1: st.image("static/11.png", use_container_width=True)
    with m2:
        st.markdown("### Step 5 (Mobile)")
        st.write("Gunakan tombol/ikon di layar (sesuai tampilan) untuk membuka panel **Pengaturan** bila diperlukan.")

# HALAMAN TENTANG
elif page == "tentang":
    st.markdown("### Pengembang Website")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image("static/fotoku.png", width=180)
    with c2:
        st.markdown("""
        **Nama** : Parveen Uzma Habidin  
        **NIM** : 535220226  
        **Jurusan** : Teknik Informatika  
        **Fakultas** : Teknik Informasi  

        **Topik Skripsi** :  
        Perencanaan Analisis Sentimen Aplikasi Sosial Media Pada Google Play Store Menggunakan Model Random Forest, Support Vector Machine dan TF-IDF
        """)
    st.markdown("---")
    a, b = st.columns(2)
    with a:
        st.markdown("### Dosen Pembimbing")
        st.image("static/pak_tri.webp", width=140)
        st.markdown("Tri Sutrisno, S.Si., M.Sc.")
    with b:
        st.markdown("### Institusi")
        st.image("static/Logo_untar.png", width=180)
        st.markdown("**Universitas Tarumanagara**")


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


# OUTPUT / HASIL
if st.session_state.results and page == "prediksi":
    items = list(st.session_state.results.items())
    cols = st.columns(len(items))
    for c, (name, dfm) in zip(cols, items):
        with c:
            st.subheader(f"Distribusi Sentimen – {name}")
            dist = dfm["pred_label"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
            st.bar_chart(dist.set_index("Sentiment"))
            st.subheader(f"Sampel Hasil Prediksi – {name}")
            st.dataframe(dfm.head(20), use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])
    with c2:
        st.download_button(
            "Download Hasil Prediksi",
            data=st.session_state.csv_pred,
            file_name=(
                f"{st.session_state.app_id}_prediksi_ulasan.xlsx"
                if st.session_state.is_combo else
                f"{st.session_state.app_id}_prediksi_ulasan.csv"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if st.session_state.is_combo else
                "text/csv"
            ),
            type="primary",
            key="dl_pred"
        )
    with c4:
        st.download_button(
            "Download Distribusi Sentimen",
            data=st.session_state.csv_dist,
            file_name=(
                f"{st.session_state.app_id}_distribusi_sentimen.xlsx"
                if st.session_state.is_combo else
                f"{st.session_state.app_id}_distribusi_sentimen.csv"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if st.session_state.is_combo else
                "text/csv"
            ),
            type="primary",
            key="dl_dist"
        )

    def rating_to_label(r):
        if pd.isna(r): return None
        return 1 if r >= 4 else 0

    metrics = []
    for name, dfm in st.session_state.results.items():
        d = dfm.copy()
        d["true_label"] = d["rating"].apply(rating_to_label)
        d = d.dropna(subset=["true_label"])
        if d.empty: continue
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(d["true_label"], d["pred"]),
            "Precision": precision_score(d["true_label"], d["pred"]),
            "Recall": recall_score(d["true_label"], d["pred"]),
            "F1-Score": f1_score(d["true_label"], d["pred"]),
        })
    if metrics:
        st.subheader("Perbandingan Metrik Evaluasi (dibanding rating bintang)")
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)
    else:
        st.info("Tidak ada metrik yang bisa dihitung.")
elif page == "prediksi" and not st.session_state.results:
    st.info("Masukkan link/package, lalu klik **Prediksi**.")
