# Laporan Proyek Machine Learning - Muhammad Hasan Fadhlillah

## Domain Proyek

Industri anggur (wine) merupakan sektor agrikultur dan ekonomi yang signifikan secara global. Kualitas anggur adalah faktor penentu utama yang mempengaruhi harga, kepuasan konsumen, dan reputasi produsen. Secara tradisional, kualitas anggur dievaluasi oleh panel pencicip ahli melalui proses sensorik yang bisa jadi subjektif, memakan waktu, dan mahal. Dengan kemajuan teknologi, terdapat kebutuhan untuk metode penilaian kualitas anggur yang lebih objektif, efisien, dan konsisten.

**Rubrik/Kriteria Tambahan (Opsional)**:

- **Mengapa dan bagaimana masalah tersebut harus diselesaikan:**
  Masalah subjektivitas dan biaya tinggi dalam penilaian kualitas anggur perlu diselesaikan untuk membantu produsen mempertahankan standar kualitas yang konsisten, mengoptimalkan proses produksi, dan memberikan informasi yang lebih transparan kepada konsumen. Pemanfaatan _machine learning_ untuk memprediksi kualitas anggur berdasarkan parameter fisikokimia dapat menjadi solusi. Model prediktif dapat menganalisis data dari tes laboratorium (seperti keasaman, kadar gula, alkohol, dll.) untuk mengklasifikasikan kualitas anggur secara otomatis. Hal ini dapat mempercepat proses evaluasi, mengurangi biaya, dan memberikan _insight_ tentang faktor-faktor apa saja yang paling berpengaruh terhadap kualitas anggur.
- **Menyertakan hasil riset terkait atau referensi:**
  1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. _Decision Support Systems, 47_(4), 547-553. (Ini adalah sumber dataset yang akan kita gunakan).
  2. Ersahin, T., Oksuz, I., & Tirkel, I. (2020). Wine Quality Prediction Using Machine Learning Algorithms. _2020 International Conference on Data Analytics for Business and Industry (ICDABI)_, 1-6. IEEE.
  3. Gupta, Y. K. (2018). Selection of important features and classification of red wine quality using machine learning techniques. _International Journal of Engineering & Technology, 7_(4.36), 704-708.
  4. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah terkait prediksi kualitas anggur merah.

### Problem Statements

- Bagaimana cara mengembangkan sistem yang objektif untuk menilai kualitas anggur merah berdasarkan data uji fisikokimia untuk mengurangi subjektivitas dan inkonsistensi dari penilai manusia?
- Faktor fisikokimia apa saja yang paling signifikan dalam menentukan kualitas anggur merah?
- Dapatkah model _machine learning_ dibangun untuk secara akurat mengklasifikasikan anggur merah ke dalam kategori kualitas (misalnya, "baik" vs "kurang baik") dengan efisiensi yang lebih tinggi daripada metode konvensional?

### Goals

- Mengembangkan model klasifikasi _machine learning_ yang mampu memprediksi kualitas anggur merah (misalnya, "baik" atau "kurang baik") berdasarkan fitur-fitur fisikokimianya.
- Mengidentifikasi fitur-fitur fisikokimia yang memiliki pengaruh paling besar terhadap kualitas anggur merah.
- Mencapai akurasi dan F1-score yang tinggi pada model prediksi kualitas anggur untuk memastikan keandalannya.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut:

  ### Solution statements

  Untuk mencapai tujuan di atas, berikut adalah pendekatan solusi yang diajukan:

  - **Menggunakan Dua Algoritma Klasifikasi:**
    - **Logistic Regression:** Sebagai model dasar (_baseline_) karena kesederhanaan dan interpretasibilitasnya.
    - **Random Forest Classifier:** Sebagai model yang lebih kompleks yang seringkali memberikan performa lebih baik pada berbagai jenis dataset dan mampu menangani hubungan non-linear serta memberikan informasi mengenai pentingnya fitur.
      Kedua model akan dievaluasi performanya menggunakan metrik seperti akurasi, presisi, recall, dan F1-score.
  - **Optimasi Model Terbaik dengan Hyperparameter Tuning:**
    Model dengan performa terbaik dari kedua algoritma di atas (berdasarkan metrik evaluasi) akan dioptimalkan lebih lanjut menggunakan teknik _hyperparameter tuning_ (misalnya, `GridSearchCV`) untuk meningkatkan kinerjanya. Solusi ini terukur karena peningkatan performa dapat dilihat dari perubahan nilai metrik evaluasi sebelum dan sesudah _tuning_.

## Data Understanding

Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Dataset yang digunakan dalam proyek ini adalah "Wine Quality Dataset," khususnya data untuk anggur merah (_red wine_). Dataset ini bersumber dari UCI Machine Learning Repository dan merupakan hasil penelitian oleh Cortez et al. (2009).

Sertakan juga sumber atau tautan untuk mengunduh dataset: [Wine Quality Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

Informasi mengenai data:

- **Jumlah Data:** Dataset anggur merah terdiri dari 1599 sampel (observasi).
- **Jumlah Fitur:** Terdapat 11 fitur input (variabel fisikokimia) dan 1 variabel output (kualitas). Semua fitur adalah numerik.
- **Kondisi Data:** Perlu diperiksa apakah ada nilai yang hilang (_missing values_) dan bagaimana distribusi masing-masing fitur serta variabel target. Variabel target 'quality' memiliki skala 3 hingga 8. Untuk tujuan klasifikasi biner, variabel ini akan diubah.

Selanjutnya uraikanlah seluruh variabel atau fitur pada data.

### Variabel-variabel pada Red Wine Quality Dataset adalah sebagai berikut:

- **`fixed acidity`**: sebagian besar asam yang terlibat dengan anggur atau tetap atau nonvolatil (tidak mudah menguap) (g/dm³).
- **`volatile acidity`**: jumlah asam asetat dalam anggur, yang pada tingkat tinggi dapat menyebabkan rasa tidak enak seperti cuka (g/dm³).
- **`citric acid`**: ditemukan dalam jumlah kecil, asam sitrat dapat menambah 'kesegaran' dan rasa pada anggur (g/dm³).
- **`residual sugar`**: jumlah gula yang tersisa setelah fermentasi berhenti (g/dm³). Jarang anggur dengan kurang dari 1 gram/liter dan anggur dengan lebih dari 45 gram/liter dianggap manis.
- **`chlorides`**: jumlah garam dalam anggur (g/dm³).
- **`free sulfur dioxide`**: SO₂ dalam bentuk bebas ada sebagai gas SO₂ terlarut atau sebagai ion bisulfit (mg/dm³).
- **`total sulfur dioxide`**: jumlah bentuk SO₂ bebas dan terikat; pada konsentrasi rendah, SO₂ sebagian besar tidak terdeteksi dalam anggur, tetapi pada konsentrasi SO₂ bebas lebih dari 50 ppm, SO₂ menjadi jelas dalam hidung dan rasa anggur (mg/dm³).
- **`density`**: massa jenis anggur, mendekati massa jenis air tergantung pada persentase alkohol dan kandungan gula (g/cm³).
- **`pH`**: menggambarkan seberapa asam atau basa suatu anggur pada skala dari 0 (sangat asam) hingga 14 (sangat basa); sebagian besar anggur berada di antara 3-4 pada skala pH.
- **`sulphates`**: aditif anggur yang dapat berkontribusi pada kadar gas sulfur dioksida (SO₂), yang bertindak sebagai antimikroba dan antioksidan (g/dm³).
- **`alcohol`**: persentase kandungan alkohol dalam anggur (vol %).
- **`quality`** (Variabel Target): skor antara 0 dan 10 (dalam dataset ini aktualnya 3-8) yang diberikan oleh penilai ahli.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.
  Tahapan EDA akan dilakukan untuk lebih memahami data, meliputi:

  - Melihat informasi dasar dataset (jumlah baris, kolom, tipe data).
  - Memeriksa _missing values_.
  - Melihat statistik deskriptif dari setiap fitur.
  - Visualisasi distribusi masing-masing fitur numerik (misalnya menggunakan histogram).
  - Visualisasi distribusi variabel target `quality` sebelum dan sesudah transformasi menjadi kategori biner.
  - Visualisasi korelasi antar fitur menggunakan _heatmap_.

  _(Contoh kode untuk EDA akan disertakan di bagian Notebook .ipynb)_

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

1.  **Memuat Dataset:** Data anggur merah (`winequality-red.csv`) dimuat ke dalam DataFrame Pandas.
2.  **Penanganan Missing Values:** Pemeriksaan dilakukan untuk nilai yang hilang. Jika ada, strategi penanganan (misalnya, imputasi atau penghapusan) akan diterapkan. (Untuk dataset ini, biasanya tidak ada _missing values_).
3.  **Transformasi Variabel Target:** Variabel `quality` (skala 3-8) akan diubah menjadi variabel biner. Anggur dengan `quality` > 5 akan dikategorikan sebagai 'baik' (1) dan `quality` &lt;= 5 sebagai 'kurang baik' (0).
4.  **Pemisahan Data:** Dataset dibagi menjadi data latih (_training set_) dan data uji (_testing set_) dengan proporsi tertentu (misalnya, 80% latih, 20% uji).
5.  **Feature Scaling (Standardization):** Fitur-fitur numerik (semua fitur input) akan distandarisasi menggunakan `StandardScaler`.

**Rubrik/Kriteria Tambahan (Opsional)**:

- **Menjelaskan proses data preparation yang dilakukan:**
  Proses dimulai dengan memuat data dan melakukan inspeksi awal termasuk pengecekan _missing values_. Variabel target 'quality' yang awalnya ordinal diubah menjadi biner untuk menyederhanakan masalah klasifikasi. Pemilihan _threshold_ (>5 untuk 'baik') didasarkan pada praktik umum dan untuk mencoba mendapatkan distribusi kelas yang lebih seimbang, meskipun ini perlu diverifikasi dengan melihat distribusi kelas setelah transformasi. Selanjutnya, data dibagi menjadi set pelatihan dan pengujian untuk evaluasi model yang tidak bias. Akhirnya, semua fitur numerik pada set pelatihan dan pengujian diskalakan menggunakan `StandardScaler`. Skaler ini di-_fit_ hanya pada data pelatihan untuk mencegah kebocoran data (_data leakage_) dari set pengujian, dan kemudian digunakan untuk mentransformasi kedua set.
- **Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut:**
  Setiap tahapan persiapan data memiliki tujuan spesifik.

  - **Transformasi Variabel Target:** Diperlukan untuk mengubah masalah menjadi format klasifikasi biner yang sesuai dengan tujuan proyek.
  - **Pemisahan Data:** Adalah praktik standar untuk mengevaluasi generalisasi model pada data yang belum pernah dilihat sebelumnya.
  - **Feature Scaling:** Penting karena beberapa algoritma (seperti Logistic Regression) sensitif terhadap skala fitur; tanpa itu, fitur dengan rentang nilai yang lebih besar dapat mendominasi proses pembelajaran. Meskipun Random Forest tidak terlalu terpengaruh oleh skala fitur, menerapkan _scaling_ tidak berdampak negatif dan memastikan konsistensi jika kita membandingkan dengan algoritma lain yang sensitif terhadap skala.

  _(Contoh kode untuk Data Preparation akan disertakan di bagian Notebook .ipynb)_

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

1.  **Pemilihan Algoritma:**
    - Logistic Regression: Sebagai _baseline model_.
    - Random Forest Classifier: Sebagai model yang lebih kompleks.
2.  **Pelatihan Model:**
    Kedua model dilatih menggunakan data latih yang telah diproses. Parameter _default_ dapat digunakan pada awalnya.
3.  **Hyperparameter Tuning:**
    Untuk model yang dipilih sebagai yang terbaik (Random Forest), dilakukan _hyperparameter tuning_ menggunakan `GridSearchCV` untuk mencari kombinasi parameter optimal yang menghasilkan performa terbaik pada data validasi silang. Parameter yang akan di-_tune_ untuk Random Forest bisa meliputi `n_estimators`, `max_depth`, `min_samples_split`, dan `min_samples_leaf`.

**Rubrik/Kriteria Tambahan (Opsional)**:

- **Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan:**
  - **Logistic Regression:**
    - _Kelebihan:_ Mudah diimplementasikan dan diinterpretasikan. Koefisien model dapat memberikan indikasi tentang pentingnya dan arah hubungan fitur dengan target. Cepat untuk dilatih, bahkan pada dataset besar. Kurang rentan terhadap _overfitting_ pada dataset dengan dimensi rendah.
    - _Kekurangan:_ Mengasumsikan hubungan linear antara fitur dan log-odds dari variabel target. Mungkin tidak bekerja dengan baik jika batas keputusan bersifat non-linear. Kurang kuat dalam menangkap interaksi kompleks antar fitur.
  - **Random Forest Classifier:**
    - _Kelebihan:_ Sangat efektif dan sering memberikan akurasi tinggi. Mampu menangani hubungan non-linear dan interaksi antar fitur secara otomatis. Kurang rentan terhadap _overfitting_ dibandingkan _single decision tree_ karena penggunaan _ensemble_ dan _bagging_. Dapat memberikan estimasi pentingnya fitur. Mampu menangani data dengan dimensi tinggi dan campuran tipe data (meskipun di sini semua numerik).
    - _Kekurangan:_ Lebih kompleks dan kurang interpretatif dibandingkan Logistic Regression (merupakan _black box model_). Membutuhkan lebih banyak waktu dan sumber daya komputasi untuk dilatih, terutama dengan jumlah _trees_ yang banyak. Parameter perlu di-_tune_ untuk performa optimal.
- **Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. Jelaskan mengapa memilih model tersebut sebagai model terbaik:**
  Setelah melatih kedua model (Logistic Regression dan Random Forest dengan parameter _default_), performa keduanya akan dibandingkan menggunakan metrik evaluasi (Akurasi, F1-score, Presisi, Recall) pada data uji. **Misalkan Random Forest menunjukkan performa yang lebih unggul secara signifikan dibandingkan Logistic Regression pada metrik-metrik kunci (terutama F1-score mengingat potensi ketidakseimbangan kelas), maka Random Forest akan dipilih sebagai model terbaik.** Alasan pemilihan adalah kemampuannya menangani kompleksitas data yang lebih baik dan potensi akurasi yang lebih tinggi.
  Selanjutnya, model Random Forest yang terpilih akan melalui proses _improvement_ dengan _hyperparameter tuning_ menggunakan `GridSearchCV`. `GridSearchCV` akan mencari kombinasi parameter terbaik (misalnya, `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`) dari _grid_ yang ditentukan dengan melakukan validasi silang. Proses ini bertujuan untuk menemukan set parameter yang memaksimalkan performa model (misalnya, F1-score) pada data yang tidak terlihat selama pelatihan setiap _fold_ validasi silang, sehingga meningkatkan kemampuan generalisasi model.

  _(Contoh kode untuk Modeling akan disertakan di bagian Notebook .ipynb)_

## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Metrik evaluasi yang digunakan adalah:

- **Akurasi (Accuracy):** Persentase prediksi yang benar dari total prediksi.
- **Presisi (Precision):** Dari semua yang diprediksi sebagai kelas positif, berapa banyak yang benar-benar positif. Penting ketika biaya _False Positive_ tinggi.
- **Recall (Sensitivity):** Dari semua yang sebenarnya kelas positif, berapa banyak yang berhasil diprediksi sebagai positif. Penting ketika biaya _False Negative_ tinggi.
- **F1-score:** Rata-rata harmonik dari Presisi dan Recall. Merupakan metrik yang baik jika ada ketidakseimbangan kelas.

**Menjelaskan hasil proyek berdasarkan metrik evaluasi:**
Hasil akan disajikan dalam bentuk tabel yang membandingkan performa Logistic Regression dan Random Forest (sebelum dan sesudah tuning) pada data uji. Misalnya:

| Model                   | Akurasi | Presisi (kelas 1) | Recall (kelas 1) | F1-score (kelas 1) |
| :---------------------- | :------ | :---------------- | :--------------- | :----------------- |
| Logistic Regression     | (nilai) | (nilai)           | (nilai)          | (nilai)            |
| Random Forest (default) | (nilai) | (nilai)           | (nilai)          | (nilai)            |
| Random Forest (tuned)   | (nilai) | (nilai)           | (nilai)          | (nilai)            |

Akan dijelaskan bagaimana model Random Forest yang telah di-_tune_ (jika ini adalah model terbaik) menunjukkan peningkatan dibandingkan versi _default_ dan model Logistic Regression. Pemilihan metrik F1-score sebagai acuan utama juga akan ditekankan jika terdapat indikasi ketidakseimbangan kelas setelah transformasi variabel target.

**Rubrik/Kriteria Tambahan (Opsional)**:

- **Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja:**

  - **Akurasi:** Mengukur proporsi total prediksi yang benar.
    - _Formula:_ `(TP + TN) / (TP + TN + FP + FN)`
    - _Cara kerja:_ Bekerja baik jika kelas seimbang, namun bisa menyesatkan jika ada ketidakseimbangan kelas.
  - **Presisi:** Untuk kelas 'baik' (1), presisi mengukur seberapa sering model benar ketika ia memprediksi anggur sebagai 'baik'.
    - _Formula:_ `TP / (TP + FP)`
    - _Cara kerja:_ Presisi tinggi berarti model jarang salah mengklasifikasikan anggur 'kurang baik' sebagai 'baik'.
  - **Recall:** Untuk kelas 'baik' (1), recall mengukur seberapa banyak dari semua anggur yang sebenarnya 'baik' berhasil diidentifikasi oleh model.
    - _Formula:_ `TP / (TP + FN)`
    - _Cara kerja:_ Recall tinggi berarti model mampu menemukan sebagian besar anggur 'baik'.
  - **F1-score:** Memberikan keseimbangan antara Presisi dan Recall.
    - _Formula:_ `2 * (Precision * Recall) / (Precision + Recall)`
    - _Cara kerja:_ Ini sangat berguna ketika distribusi kelas tidak seimbang karena ia memperhitungkan kedua jenis kesalahan (FP dan FN). F1-score yang tinggi menunjukkan bahwa model memiliki presisi dan recall yang baik.

  Dimana:

  - TP (True Positive): Sampel positif yang diprediksi benar.
  - TN (True Negative): Sampel negatif yang diprediksi benar.
  - FP (False Positive): Sampel negatif yang salah diprediksi sebagai positif (Type I Error).
  - FN (False Negative): Sampel positif yang salah diprediksi sebagai negatif (Type II Error).

  Metrik evaluasi ini dipilih karena memberikan gambaran komprehensif tentang kinerja model klasifikasi, terutama dalam konteks di mana identifikasi anggur 'baik' dan 'kurang baik' sama-sama penting, dan potensi ketidakseimbangan kelas perlu diperhatikan.

  _(Contoh kode untuk Evaluation akan disertakan di bagian Notebook .ipynb)_

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
