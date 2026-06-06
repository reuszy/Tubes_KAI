# Dashboard Klasifikasi Keluhan IndiHome

Dashboard interaktif dengan Streamlit untuk visualisasi dan demo model SVM
hasil klasifikasi keluhan pelanggan IndiHome.

## Persiapan

### 1. Install Library

Buka terminal di folder dashboard ini, lalu jalankan:

```bash
pip install streamlit pandas numpy scikit-learn nltk Sastrawi matplotlib seaborn wordcloud joblib
```

### 2. Pastikan 3 file ini ada di folder yang sama dengan dashboard.py:

- `model_svm_indihome.pkl` (hasil dari notebook bagian 8)
- `tfidf_vectorizer.pkl` (hasil dari notebook bagian 8)
- File CSV dataset Anda (misal `indihome_care_..._Label_FIX_2.csv`)

Kalau file model belum ada, jalankan dulu notebook `klasifikasi_keluhan_v3.ipynb`
sampai akhir agar file `.pkl` ter-generate.

## Cara Menjalankan

Buka terminal di folder dashboard, lalu ketik:

```bash
streamlit run dashboard.py
```

Otomatis browser akan terbuka di `http://localhost:8501`.

Untuk berhenti, tekan `Ctrl + C` di terminal.

## Fitur Dashboard

### Tab 1: Beranda

- Metric cards: total tweet, jumlah kategori, periode, kategori dominan
- Grafik distribusi kategori
- WordCloud keseluruhan
- Insight singkat

### Tab 2: Prediksi Keluhan

- Input text area untuk tweet baru
- Tombol contoh tweet siap pakai
- Hasil prediksi dengan confidence score
- Bar chart probabilitas semua kategori
- Detail preprocessing teks

### Tab 3: Eksplorasi Data

- Filter berdasarkan kategori
- Statistik per kategori
- Top 10 kata dominan per kategori
- WordCloud per kategori
- Sampel tweet (tabel)

### Tab 4: Performa Model

- Metric cards: akurasi, precision, recall, F1
- Confusion matrix (heatmap)
- Tabel metrik per kategori
- Interpretasi hasil
