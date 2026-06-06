import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(
    page_title="Dashboard Klasifikasi Keluhan IndiHome",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    model = joblib.load('Model/model_svm_indihome.pkl')
    tfidf = joblib.load('Model/tfidf_vectorizer.pkl')
    return model, tfidf

@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename, encoding='utf-8-sig')

    if 'tanggal' in df.columns:
        df['tanggal_parsed'] = pd.to_datetime(df['tanggal'], errors='coerce')
    return df

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

KAMUS_NORMALISASI = {
    'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'kagak': 'tidak', 'engga': 'tidak',
    'bgt': 'banget', 'bngt': 'banget', 'bener': 'benar',
    'gimana': 'bagaimana', 'gmn': 'bagaimana', 'knp': 'kenapa',
    'napa': 'kenapa', 'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
    'blm': 'belum', 'belom': 'belum', 'lg': 'lagi', 'lgi': 'lagi',
    'aja': 'saja', 'aj': 'saja', 'dr': 'dari', 'dri': 'dari',
    'sm': 'sama', 'tp': 'tapi', 'tpi': 'tapi', 'krn': 'karena',
    'karna': 'karena', 'utk': 'untuk', 'buat': 'untuk', 'jd': 'jadi',
    'jdi': 'jadi', 'sy': 'saya', 'gw': 'saya', 'gua': 'saya',
    'gue': 'saya', 'lu': 'kamu', 'lo': 'kamu', 'min': 'admin',
    'kak': 'kakak', 'inet': 'internet', 'lemot': 'lambat',
    'lelet': 'lambat', 'pls': 'tolong', 'plis': 'tolong',
    'gajelas': 'tidak jelas', 'gabisa': 'tidak bisa', 'gaada': 'tidak ada',
}

def preprocess_text(teks, stemmer):
    if not isinstance(teks, str):
        return ""
    # Lowercase
    teks = teks.lower()
    # Hapus URL, mention, hashtag, angka, tanda baca, non-ASCII
    teks = re.sub(r'http\S+|www\.\S+', ' ', teks)
    teks = re.sub(r'@\w+', ' ', teks)
    teks = re.sub(r'#', ' ', teks)
    teks = re.sub(r'\d+', ' ', teks)
    teks = teks.translate(str.maketrans('', '', string.punctuation))
    teks = re.sub(r'[^\x00-\x7f]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    # Normalisasi
    kata_list = teks.split()
    kata_list = [KAMUS_NORMALISASI.get(k, k) for k in kata_list]
    # Stopword removal
    stop_id = set(stopwords.words('indonesian')) - {'tidak', 'belum', 'kurang', 'sulit'}
    kata_list = [k for k in kata_list if k not in stop_id]
    # Stemming
    kata_list = [stemmer.stem(k) for k in kata_list]
    return ' '.join(kata_list)


st.title("Dashboard Klasifikasi Keluhan Pelanggan IndiHome")
st.caption("Analisis dan Klasifikasi Otomatis Topik Keluhan pada Media Sosial X (Twitter) Menggunakan Support Vector Machine")

# Sidebar
with st.sidebar:
    st.header("Konfigurasi")
    
    nama_file = st.text_input(
        "Nama file dataset CSV:",
        value="keluhan_indihome.csv",
        help="File CSV harus ada di folder yang sama dengan dashboard.py"
    )
    
    st.divider()
    st.markdown("""
    Dashboard ini adalah hasil tugas besar mata kuliah Kecerdasan AI dan Penerapannya.
    
    **Metode:** Support Vector Machine  
    **Sumber data:** Twitter/X (@IndiHomeCare)  
    **Kategori:** 5 topik keluhan
    """)

try:
    model, tfidf = load_model()
    df = load_data(nama_file)
    stemmer = load_stemmer()
    LABEL_VALID = sorted(df['label'].dropna().unique())
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e}")
    st.info("Pastikan file model (.pkl) dan dataset (.csv) ada di folder yang sama dengan dashboard.py")
    st.stop()
except Exception as e:
    st.error(f"Terjadi error: {e}")
    st.stop()

#Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "Beranda",
    "Prediksi Keluhan",
    "Eksplorasi Data",
    "Performa Model"
])


# TAB 1: BERANDA
with tab1:
    st.header("Ringkasan Dataset")
    
    # Metric cards di atas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tweet", f"{len(df):,}")
    with col2:
        st.metric("Jumlah Kategori", df['label'].nunique())
    with col3:
        if 'tanggal_parsed' in df.columns:
            periode = f"{df['tanggal_parsed'].min().year}–{df['tanggal_parsed'].max().year}"
            st.metric("Periode Data", periode)
        else:
            st.metric("Periode Data", "—")
    with col4:
        kategori_dominan = df['label'].value_counts().idxmax()
        st.metric("Kategori Dominan", kategori_dominan)
    
    st.divider()
    
    # 2 kolom: distribusi + wordcloud
    col_kiri, col_kanan = st.columns([1, 1])
    
    with col_kiri:
        st.subheader("Distribusi Kategori")
        dist = df['label'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        warna = sns.color_palette('Set2', len(dist))
        bars = ax.barh(dist.index, dist.values, color=warna, edgecolor='black')
        ax.set_xlabel('Jumlah Tweet')
        ax.set_title('Jumlah Tweet per Kategori', fontweight='bold')
        for i, v in enumerate(dist.values):
            ax.text(v + 5, i, str(v), va='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_kanan:
        st.subheader("WordCloud Keseluruhan")
        # Gabung tweet (yang masih raw, untuk kecepatan tidak preprocess ulang)
        # Sebagai approximation, kita ambil kata > 4 karakter untuk hindari noise
        teks_sample = ' '.join(df['tweet'].astype(str).str.lower())
        teks_sample = re.sub(r'http\S+|@\w+|#|\d+', ' ', teks_sample)
        teks_sample = re.sub(r'[^\w\s]', ' ', teks_sample)
        
        # Stopwords ringan untuk display saja
        stop_simple = set(stopwords.words('indonesian'))
        teks_words = [w for w in teks_sample.split() if len(w) > 3 and w not in stop_simple]
        teks_clean = ' '.join(teks_words)
        
        if teks_clean.strip():
            wc = WordCloud(
                width=800, height=500,
                background_color='white',
                colormap='viridis',
                max_words=80,
                collocations=False
            ).generate(teks_clean)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
    
    st.divider()
    
    # Insight singkat
    st.subheader("Insight Singkat")
    persen_dominan = dist.iloc[0] / dist.sum() * 100
    st.info(f"""
    - Dataset berisi **{len(df):,} tweet keluhan** dari pelanggan IndiHome
    - Kategori **{dist.index[0]}** mendominasi dengan **{persen_dominan:.1f}%** dari total
    - Distribusi kategori tidak seimbang, ditangani dengan `class_weight='balanced'` saat training
    - Periode data mencakup beberapa tahun sehingga representatif
    """)


# TAB 2: PREDIKSI KELUHAN
with tab2:
    st.header("Prediksi Kategori Keluhan")
    st.markdown("Masukkan tweet keluhan untuk diprediksi kategorinya oleh model SVM.")
    
    # Input area
    contoh_tweet = [
        "Internet saya putus-putus dari kemarin tolong diperbaiki",
        "Kenapa tagihan bulan ini naik tiba-tiba",
        "DM saya dari kemarin tidak dibalas admin",
        "Mau berhenti berlangganan indihome gimana caranya",
        "Wifi saya mati total dari pagi sampai sekarang"
    ]
    
    col_input, col_contoh = st.columns([2, 1])
    
    with col_contoh:
        st.markdown("**Contoh tweet (klik untuk pakai):**")
        for c in contoh_tweet:
            if st.button(f"{c[:40]}...", key=f"btn_{c[:20]}", use_container_width=True):
                st.session_state['tweet_input'] = c
    
    with col_input:
        tweet_input = st.text_area(
            "Ketik tweet keluhan di sini:",
            value=st.session_state.get('tweet_input', ''),
            height=150,
            placeholder="Contoh: Internet saya lemot banget dari kemarin..."
        )
        
        tombol_prediksi = st.button("Prediksi Kategori", type="primary", use_container_width=True)
    
    if tombol_prediksi:
        if not tweet_input.strip():
            st.warning("Silakan masukkan tweet terlebih dahulu!")
        else:
            with st.spinner("Memproses prediksi"):
                # Preprocessing
                teks_bersih = preprocess_text(tweet_input, stemmer)
                
                if not teks_bersih:
                    st.error("Teks terlalu pendek setelah preprocessing, tidak bisa diprediksi.")
                else:
                    # Prediksi
                    vektor = tfidf.transform([teks_bersih])
                    prediksi = model.predict(vektor)[0]
                    
                    # Probabilitas (decision_function untuk SVM)
                    decision = model.decision_function(vektor)[0]
                    kelas = model.classes_
                    
                    # Konversi decision score jadi probabilitas (softmax)
                    exp_scores = np.exp(decision - np.max(decision))
                    probs = exp_scores / exp_scores.sum()
                    
                    #  HASIL UTAMA 
                    st.divider()
                    st.subheader("Hasil Prediksi")
                    
                    confidence = probs[list(kelas).index(prediksi)] * 100
                    
                    col_hasil1, col_hasil2 = st.columns([1, 1])
                    
                    with col_hasil1:
                        st.markdown(f"### Kategori: **{prediksi}**")
                        st.metric("Confidence Score", f"{confidence:.1f}%")
                        
                        # Interpretasi confidence
                        if confidence > 70:
                            st.success("Model sangat yakin dengan prediksi ini")
                        elif confidence > 50:
                            st.info("Model cukup yakin dengan prediksi ini")
                        else:
                            st.warning("Model kurang yakin, pertimbangkan top 3")
                    
                    with col_hasil2:
                        # Bar chart probabilitas semua kategori
                        prob_df = pd.DataFrame({
                            'Kategori': kelas,
                            'Probabilitas': probs * 100
                        }).sort_values('Probabilitas', ascending=True)
                        
                        fig, ax = plt.subplots(figsize=(7, 4))
                        warna = ['#3498DB' if k == prediksi else '#BDC3C7' 
                                for k in prob_df['Kategori']]
                        ax.barh(prob_df['Kategori'], prob_df['Probabilitas'],
                               color=warna, edgecolor='black')
                        ax.set_xlabel('Probabilitas (%)')
                        ax.set_title('Probabilitas per Kategori', fontweight='bold')
                        for i, v in enumerate(prob_df['Probabilitas']):
                            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    #  DETAIL TEKS 
                    with st.expander("Lihat detail preprocessing"):
                        st.markdown("**Teks asli:**")
                        st.code(tweet_input)
                        st.markdown("**Setelah preprocessing:**")
                        st.code(teks_bersih if teks_bersih else "(kosong)")


# TAB 3: EKSPLORASI DATA
with tab3:
    st.header("Eksplorasi Data per Kategori")
    
    # Filter kategori
    kategori_pilihan = st.selectbox(
        "Pilih kategori untuk dieksplorasi:",
        options=['Semua'] + LABEL_VALID,
        index=0
    )
    
    # Filter data
    if kategori_pilihan == 'Semua':
        df_filter = df.copy()
    else:
        df_filter = df[df['label'] == kategori_pilihan].copy()
    
    st.markdown(f"**Menampilkan {len(df_filter)} tweet** dalam kategori **{kategori_pilihan}**")
    st.divider()
    
    # Layout: stats kiri, wordcloud kanan
    col_stats, col_wc = st.columns([1, 1])
    
    with col_stats:
        st.subheader("Statistik")
        if len(df_filter) > 0:
            df_filter['panjang'] = df_filter['tweet'].astype(str).str.len()
            df_filter['kata'] = df_filter['tweet'].astype(str).str.split().str.len()
            
            st.metric("Rata-rata panjang tweet", f"{df_filter['panjang'].mean():.0f} karakter")
            st.metric("Rata-rata jumlah kata", f"{df_filter['kata'].mean():.0f} kata")
            
            # Top 10 kata di kategori ini
            st.markdown("**Top 10 kata sering muncul:**")
            teks_kat = ' '.join(df_filter['tweet'].astype(str).str.lower())
            teks_kat = re.sub(r'http\S+|@\w+|#|\d+', ' ', teks_kat)
            teks_kat = re.sub(r'[^\w\s]', ' ', teks_kat)
            stop_simple = set(stopwords.words('indonesian'))
            kata_list = [w for w in teks_kat.split() if len(w) > 3 and w not in stop_simple]
            top10 = Counter(kata_list).most_common(10)
            
            if top10:
                top10_df = pd.DataFrame(top10, columns=['Kata', 'Frekuensi'])
                st.dataframe(top10_df, use_container_width=True, hide_index=True)
    
    with col_wc:
        st.subheader("WordCloud")
        if len(df_filter) > 0:
            teks_wc = ' '.join(df_filter['tweet'].astype(str).str.lower())
            teks_wc = re.sub(r'http\S+|@\w+|#|\d+', ' ', teks_wc)
            teks_wc = re.sub(r'[^\w\s]', ' ', teks_wc)
            stop_simple = set(stopwords.words('indonesian'))
            words = [w for w in teks_wc.split() if len(w) > 3 and w not in stop_simple]
            teks_clean = ' '.join(words)
            
            if teks_clean.strip():
                wc = WordCloud(
                    width=700, height=400,
                    background_color='white',
                    colormap='plasma',
                    max_words=60,
                    collocations=False
                ).generate(teks_clean)
                
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Tidak cukup kata untuk WordCloud")
    
    st.divider()
    
    # Tabel sampel tweet
    st.subheader("Sampel Tweet")
    jumlah_sampel = st.slider("Jumlah tweet ditampilkan:", 5, 50, 15)
    
    kolom_tampil = ['label', 'tweet']
    if 'tanggal' in df_filter.columns:
        kolom_tampil = ['tanggal', 'label', 'tweet']
    
    sampel = df_filter[kolom_tampil].head(jumlah_sampel)
    if 'tanggal' in sampel.columns:
        sampel['tanggal'] = pd.to_datetime(sampel['tanggal'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    st.dataframe(sampel, use_container_width=True, hide_index=True)


# TAB 4: PERFORMA MODEL
with tab4:
    st.header("Performa Model SVM")
    st.markdown("Evaluasi model pada data testing (20% dari dataset).")
    
    # Lakukan preprocessing untuk evaluasi (dengan progress bar)
    @st.cache_data
    def hitung_evaluasi(_df, _stemmer):
        """Hitung metrik evaluasi - di-cache agar tidak diulang."""
        # Preprocess semua tweet (atau sample untuk kecepatan)
        teks_bersih = _df['tweet'].apply(lambda t: preprocess_text(t, _stemmer))
        df_eval = pd.DataFrame({
            'teks': teks_bersih,
            'label': _df['label']
        })
        df_eval = df_eval[df_eval['teks'].str.strip() != ''].reset_index(drop=True)
        
        # Transform & split (random_state sama dengan training)
        X = tfidf.transform(df_eval['teks'])
        y = df_eval['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        y_pred = model.predict(X_test)
        akurasi = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        labels = sorted(y.unique())
        
        return akurasi, report, cm, labels
    
    with st.spinner("Menghitung metrik evaluasi (preprocessing data)..."):
        akurasi, report, cm, label_list = hitung_evaluasi(df, stemmer)
    
    # Metric cards untuk metrik utama
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Akurasi", f"{akurasi*100:.2f}%")
    with col_m2:
        st.metric("Macro F1-Score", f"{report['macro avg']['f1-score']*100:.2f}%")
    with col_m3:
        st.metric("Macro Precision", f"{report['macro avg']['precision']*100:.2f}%")
    with col_m4:
        st.metric("Macro Recall", f"{report['macro avg']['recall']*100:.2f}%")
    
    st.divider()
    
    # Confusion Matrix + Tabel Metrik
    col_cm, col_table = st.columns([1, 1])
    
    with col_cm:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_list, yticklabels=label_list, ax=ax,
                   cbar_kws={'label': 'Jumlah Prediksi'})
        ax.set_ylabel('Label Sebenarnya')
        ax.set_xlabel('Label Prediksi')
        ax.set_title('Confusion Matrix', fontweight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("Diagonal = prediksi benar. Di luar diagonal = kesalahan.")
    
    with col_table:
        st.subheader("Metrik per Kategori")
        # Buat tabel rapi dari classification_report
        rows = []
        for label in label_list:
            if label in report:
                rows.append({
                    'Kategori': label,
                    'Precision': f"{report[label]['precision']*100:.1f}%",
                    'Recall': f"{report[label]['recall']*100:.1f}%",
                    'F1-Score': f"{report[label]['f1-score']*100:.1f}%",
                    'Support': int(report[label]['support'])
                })
        
        metrik_df = pd.DataFrame(rows)
        st.dataframe(metrik_df, use_container_width=True, hide_index=True)
        
        st.markdown("**Penjelasan metrik:**")
        st.caption("""
        - **Precision**: dari yang diprediksi kategori X, berapa % yang benar
        - **Recall**: dari yang sebenarnya kategori X, berapa % berhasil ditemukan
        - **F1-Score**: rata-rata harmonik precision dan recall
        - **Support**: jumlah tweet di kategori tersebut (data testing)
        """)
    
    st.divider()
    
    # Interpretasi & insight
    st.subheader("Interpretasi Hasil")
    
    # Cari kategori dengan F1 terbaik dan terburuk
    f1_per_kelas = {l: report[l]['f1-score'] for l in label_list if l in report}
    best_kat = max(f1_per_kelas, key=f1_per_kelas.get)
    worst_kat = min(f1_per_kelas, key=f1_per_kelas.get)
    
    st.markdown(f"""
    - Model mencapai akurasi keseluruhan **{akurasi*100:.2f}%**
    - Kategori dengan performa terbaik: **{best_kat}** (F1: {f1_per_kelas[best_kat]*100:.1f}%)
    - Kategori dengan performa terlemah: **{worst_kat}** (F1: {f1_per_kelas[worst_kat]*100:.1f}%)
    - Performa rendah biasanya disebabkan: jumlah data sedikit, batas kategori ambigu,
      atau pola bahasa yang mirip dengan kategori lain
    """)
