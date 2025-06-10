# Red Wine Quality Prediction 🍷✨

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white" alt="Python Version">Add commentMore actions
  <img src="https://img.shields.io/badge/Scikit--learn-0.24%2B-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-1.1%2B-green?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-1.19%2B-blueviolet?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-3.3%2B-yellowgreen?logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-0.11%2B-purple?logo=seaborn&logoColor=white" alt="Seaborn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" alt="Jupyter Notebook">
</p>

## Ringkasan Proyek

Proyek ini bertujuan untuk mengembangkan model _machine learning_ yang mampu memprediksi kualitas anggur merah (_red wine_) berdasarkan atribut fisikokimianya. Secara tradisional, evaluasi kualitas anggur bergantung pada penilaian sensorik oleh para ahli yang bisa jadi subjektif, mahal, dan memakan waktu. Proyek ini menawarkan solusi berbasis data untuk menciptakan sistem penilaian yang lebih objektif, efisien, dan konsisten.

Dengan menggunakan dataset dari UCI Machine Learning Repository, proyek ini mencakup analisis data eksplorasi (EDA), pra-pemrosesan data, pemodelan dengan beberapa algoritma klasifikasi, evaluasi performa, hingga identifikasi faktor-faktor kimia yang paling berpengaruh terhadap kualitas anggur.

## Business Understanding

### Latar Belakang Masalah

Industri anggur memerlukan metode penilaian kualitas yang konsisten untuk menjaga reputasi produsen, mengoptimalkan produksi, dan memberikan transparansi kepada konsumen. Ketergantungan pada panel pencicip manusia menimbulkan tantangan berupa subjektivitas, biaya tinggi, dan waktu yang lama. Oleh karena itu, diperlukan sebuah sistem otomatis yang dapat mengklasifikasikan kualitas anggur secara objektif berdasarkan data uji laboratorium.

### Problem Statements

- Bagaimana cara mengembangkan sistem yang objektif untuk menilai kualitas anggur merah berdasarkan data uji fisikokimia guna mengurangi subjektivitas?
- Faktor fisikokimia apa saja yang paling signifikan dalam menentukan kualitas anggur merah?
- Dapatkah model _machine learning_ dibangun untuk secara akurat mengklasifikasikan anggur merah ke dalam kategori kualitas ("Baik" vs "Tidak Baik") dengan efisiensi yang lebih tinggi?

### Goals

- Mengembangkan dan membandingkan beberapa model klasifikasi _machine learning_ untuk memilih model terbaik dalam memprediksi kualitas anggur.
- Mengidentifikasi fitur-fitur fisikokimia yang memiliki pengaruh paling signifikan terhadap kualitas anggur menggunakan model terpilih.
- Mencapai F1-score setinggi mungkin untuk memastikan model andal dalam menangani potensi ketidakseimbangan kelas.

### Rencana Solusi

1. **Pemodelan dengan Tiga Algoritma:**
   - **Logistic Regression:** Sebagai model _baseline_ yang sederhana dan mudah diinterpretasikan.
     - **Random Forest Classifier:** Sebagai model _ensemble_ yang kuat untuk menangani hubungan non-linear.
     - **Support Vector Machine (SVM):** Sebagai model yang efektif dalam ruang berdimensi tinggi.
2. **Evaluasi dan Pemilihan Model:** Ketiga model dievaluasi menggunakan metrik Akurasi, Presisi, Recall, dan F1-Score. Model dengan F1-score tertinggi pada data uji akan dipilih sebagai yang terbaik.
3. **Identifikasi Fitur Penting:** Teknik _Permutation Importance_ diterapkan pada model terbaik untuk mengukur pengaruh setiap fitur fisikokimia.

## Data Understanding

Dataset yang digunakan adalah **"Red Wine Quality Dataset"** dari UCI Machine Learning Repository.

- **Sumber:** [Wine Quality Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- **Struktur:** 1599 sampel, 11 fitur input (numerik), dan 1 variabel target (`quality`).
- **Kualitas Data:** Tidak ada nilai yang hilang (_missing values_), namun ditemukan dan ditangani 240 baris duplikat.
- **Variabel Target:** `quality` (skala 3-8) ditransformasi menjadi `quality_category` biner: - **0 (Tidak Baik):** jika `quality` \<= 5 - **1 (Baik):** jika `quality` \> 5

### Atribut Dataset

1. `fixed acidity`
2. `volatile acidity`
3. `citric acid`
4. `residual sugar`
5. `chlorides`
6. `free sulfur dioxide`
7. `total sulfur dioxide`
8. `density`
9. `pH`
10. `sulphates`
11. `alcohol`
12. `quality` (Target Asli)

### Temuan dari Analisis Data Eksplorasi (EDA)

- **Distribusi Kelas:** Cukup seimbang setelah transformasi (855 'Baik', 744 'Tidak Baik').
- **Distribusi Fitur:** Sebagian besar fitur memiliki distribusi miring (_skewed_).
- **Outliers:** Terdeteksi banyak _outlier_ pada hampir semua fitur, yang menandakan perlunya penanganan khusus.
- **Korelasi:** `alcohol` memiliki korelasi positif terkuat (0.48) dengan `quality`, sedangkan `volatile acidity` memiliki korelasi negatif terkuat (-0.39).

## Data Preparation

Tahapan persiapan data dilakukan secara berurutan untuk memastikan kualitas data sebelum pemodelan:

1. **Transformasi Variabel Target:** Mengubah `quality` menjadi `quality_category` biner (0/1).
2. **Penanganan Duplikat:** Menghapus 240 baris data duplikat. Ukuran dataset menjadi 1359 sampel.
3. **Penanganan Outlier:** Menangani _outlier_ pada semua fitur input menggunakan metode IQR, di mana nilai di luar batas digantikan dengan nilai batas atas/bawah (_capping_).
4. **Pemisahan Data:** Membagi dataset menjadi 80% data latih dan 20% data uji, dengan stratifikasi pada variabel target untuk menjaga proporsi kelas.
5. **Standardisasi Fitur:** Menskalakan semua fitur input menggunakan `StandardScaler` yang dilatih hanya pada data latih untuk mencegah kebocoran data.

## Modeling

Model terbaik dipilih berdasarkan F1-Score pada data uji, karena metrik ini menyeimbangkan Presisi dan Recall.

| Model                   | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Keterangan                                                                       |
| :---------------------- | :------------: | :-----------: | :------------: | :---------: | :-----------: | :------------------------------------------------------------------------------- |
| Logistic Regression     |     0.7470     |    0.7243     |     0.7257     |   0.7243    |    0.7245     | Model _baseline_.                                                                |
| Random Forest (Default) |     1.0000     |    0.7610     |     0.7624     |   0.7610    |    0.7612     | Performa baik, namun menunjukkan _overfitting_ yang signifikan.                  |
| **SVM (Default)**       |   **0.8050**   |  **0.7721**   |   **0.7745**   | **0.7721**  |  **0.7722**   | **Model Terbaik:** Performa generalisasi paling seimbang dan F1-Score tertinggi. |
| SVM (Tuned)             |     0.7893     |    0.7537     |     0.7546     |   0.7537    |    0.7539     | Performa menurun pada data uji setelah _hyperparameter tuning_.                  |

(Metrik Precision, Recall, dan F1-Score adalah 'weighted average')

**Model Akhir yang Dipilih:** **Support Vector Machine (SVM) dengan parameter default** dipilih sebagai solusi akhir. Meskipun _hyperparameter tuning_ dengan `GridSearchCV` telah dilakukan, model SVM dengan parameter default menunjukkan F1-Score dan kemampuan generalisasi yang lebih unggul pada data uji.

## Evaluation

### Faktor Paling Berpengaruh (Feature Importance)

Berdasarkan hasil **Permutation Importance** yang diterapkan pada model SVM terbaik, faktor fisikokimia yang paling signifikan dalam menentukan kualitas anggur merah adalah:

1. **`alcohol`**: Kadar alkohol adalah prediktor terkuat.
2. **`sulphates`**: Penting untuk preservasi dan kualitas.
3. **`volatile acidity`**: Keasaman yang tidak diinginkan menjadi penentu utama kualitas rendah.
4. **`total sulfur dioxide`**: Konsentrasi sulfur dioksida juga memainkan peran penting.

Hasil ini konsisten dengan analisis korelasi pada tahap EDA, memberikan keyakinan bahwa kesimpulan yang ditarik bersifat solid.

### Metrik Evaluasi

Proyek ini menggunakan metrik standar klasifikasi untuk evaluasi:

- **Akurasi (Accuracy):** Persentase total prediksi yang benar.
- **Presisi (Precision):** Kemampuan model untuk tidak salah melabeli sampel negatif sebagai positif. $$\text{Presisi} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Recall (Sensitivity):** Kemampuan model untuk menemukan semua sampel positif yang sebenarnya. $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **F1-score:** Rata-rata harmonik dari Presisi dan Recall, menjadi metrik utama. $$\text{F1-Score} = 2 \times \frac{\text{Presisi} \times \text{Recall}}{\text{Presisi} + \text{Recall}}$$

## 🚀 How to Run

1. **Clone repositori:**

   ```bash
   git clone https://github.com/your-username/red-wine-quality-predictor.git
   cd red-wine-quality-predictor
   ```

2. **Buat lingkungan virtual (disarankan):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Pada Windows:
   venv\Scripts\activate
   ```

3. **Instal dependensi:**

   ```bash
   pip install -r requirements.txt

   (Anda perlu membuat file `requirements.txt` dengan menjalankan `pip freeze > requirements.txt` setelah menginstal semua pustaka yang diperlukan: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter).

   ```

4. **Unduh dataset:**

   Pastikan `winequality-red.csv` berada di direktori proyek atau perbarui path di dalam notebook. Anda dapat mengunduhnya dari [sini]
   (<https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv>).

5. **Jalankan Jupyter Notebook:**

   ```bash
   jupyter notebook Notebook.ipynb
   ```
