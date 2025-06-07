# Laporan Proyek Machine Learning - Muhammad Hasan Fadhlillah

## Domain Proyek

Industri anggur (wine) merupakan sektor agrikultur dan ekonomi yang signifikan secara global. Kualitas anggur adalah faktor penentu utama yang mempengaruhi harga, kepuasan konsumen, dan reputasi produsen. Secara tradisional, kualitas anggur dievaluasi oleh panel pencicip ahli melalui proses sensorik yang bisa jadi subjektif, memakan waktu, dan mahal. Dengan kemajuan teknologi, terdapat kebutuhan untuk metode penilaian kualitas anggur yang lebih objektif, efisien, dan konsisten.

- **Mengapa dan bagaimana masalah tersebut harus diselesaikan:**
  Masalah subjektivitas dan biaya tinggi dalam penilaian kualitas anggur perlu diselesaikan untuk membantu produsen mempertahankan standar kualitas yang konsisten, mengoptimalkan proses produksi, dan memberikan informasi yang lebih transparan kepada konsumen. Pemanfaatan machine learning untuk memprediksi kualitas anggur berdasarkan parameter fisikokimia dapat menjadi solusi. Model prediktif dapat menganalisis data dari tes laboratorium (seperti keasaman, kadar gula, alkohol, dll.) untuk mengklasifikasikan kualitas anggur secara otomatis. Hal ini dapat mempercepat proses evaluasi, mengurangi biaya, dan memberikan insight tentang faktor-faktor apa saja yang paling berpengaruh terhadap kualitas anggur.
- **Menyertakan hasil riset terkait atau referensi:**
  1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. _Decision Support Systems, 47_(4), 547-553. (Ini adalah sumber dataset yang akan kita gunakan).
  2. Ersahin, T., Oksuz, I., & Tirkel, I. (2020). Wine Quality Prediction Using Machine Learning Algorithms. _2020 International Conference on Data Analytics for Business and Industry (ICDABI)_, 1-6. IEEE.
  3. Gupta, Y. K. (2018). Selection of important features and classification of red wine quality using machine learning techniques. _International Journal of Engineering & Technology, 7_(4.36), 704-708.

## Business Understanding

Pada bagian ini, dijelaskan proses klarifikasi masalah terkait prediksi kualitas anggur merah.

### Problem Statements

- Bagaimana cara mengembangkan sistem yang objektif untuk menilai kualitas anggur merah berdasarkan data uji fisikokimia untuk mengurangi subjektivitas dan inkonsistensi dari penilai manusia?
- Faktor fisikokimia apa saja yang paling signifikan dalam menentukan kualitas anggur merah?
- Dapatkah model _machine learning_ dibangun untuk secara akurat mengklasifikasikan anggur merah ke dalam kategori kualitas ("Baik" vs "Tidak Baik") dengan efisiensi yang lebih tinggi daripada metode konvensional?

### Goals

- Mengembangkan beberapa model klasifikasi _machine learning_ untuk memprediksi kualitas anggur merah ("Baik" atau "Tidak Baik") dan memilih model terbaik.
- Mengidentifikasi fitur-fitur fisikokimia yang memiliki pengaruh paling besar terhadap kualitas anggur merah menggunakan model terbaik tersebut.
- Mencapai F1-score setinggi mungkin pada model prediksi kualitas anggur untuk memastikan keandalannya dalam menangani potensi ketidakseimbangan kelas.

  ### Solution statements

  Untuk mencapai tujuan di atas, berikut adalah pendekatan solusi yang diajukan:

  - **Menggunakan Tiga Algoritma Klasifikasi:**
    - **Logistic Regression:** Sebagai model dasar (_baseline_) karena kesederhanaan dan interpretasibilitasnya.
    - **Random Forest Classifier:** Sebagai model _ensemble_ yang kuat dan seringkali memberikan performa yang baik.
    - **Support Vector Machine (SVM):** Sebagai model kuat lainnya yang efektif dalam ruang berdimensi tinggi.
      Ketiga model akan dievaluasi performanya menggunakan metrik Akurasi, Presisi, Recall, dan F1-score untuk memilih model dengan kinerja terbaik.
  - **Optimasi Model Terbaik dengan Hyperparameter Tuning:**
    Model dengan performa terbaik dari ketiga algoritma di atas (berdasarkan metrik F1-score) akan dioptimalkan lebih lanjut menggunakan teknik `GridSearchCV`. Solusi ini terukur karena peningkatan (atau perubahan) performa dapat dilihat dari perbandingan nilai metrik evaluasi sebelum dan sesudah _tuning_.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Wine Quality Dataset," khususnya data untuk anggur merah (_red wine_). Dataset ini bersumber dari UCI Machine Learning Repository dan merupakan hasil penelitian oleh Cortez et al. (2009).

**Sumber Data:** [Wine Quality Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) (digunakan dalam notebook via path lokal `./datasets/winequality-red.csv`).

**Informasi mengenai data:**

- **Jumlah Data:** Dataset anggur merah terdiri dari 1599 sampel.
- **Jumlah Fitur:** Terdapat 11 fitur input (variabel fisikokimia) dan 1 variabel output (kualitas). Semua fitur adalah numerik.
- **Kualitas Data:** Tidak ditemukan nilai yang hilang (_missing values_), namun ditemukan adanya 240 baris duplikat.
- **Variabel Target:** Variabel target `quality` memiliki skala 3 hingga 8. Untuk tujuan klasifikasi biner, variabel ini akan diubah menjadi `quality_category` (0 untuk 'Tidak Baik', 1 untuk 'Baik').

### Variabel-variabel pada Red Wine Quality Dataset

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
- **`quality`** (Variabel Target): skor antara 3 dan 8 yang diberikan oleh penilai ahli.

**Rubrik/Kriteria Tambahan:**

Tahapan EDA telah dilakukan untuk lebih memahami data, meliputi:

- Melihat informasi dasar dataset (jumlah baris, kolom, tipe data).
- Memeriksa _missing values_.
- Melihat statistik deskriptif dari setiap fitur.
- **Distribusi Variabel Target:** Variabel `quality` diubah menjadi biner (`quality_category`), dengan 744 sampel masuk kategori 'Tidak Baik' (≤5) dan 855 sampel 'Baik' (>5). Distribusi kelas ini cukup seimbang.
- **Distribusi Fitur Numerik:** Visualisasi menggunakan histogram menunjukkan bahwa banyak fitur memiliki distribusi yang miring (_skewed_), seperti `residual sugar`, `chlorides`, dan `sulphates`.
- **Deteksi Outlier:** Boxplot menunjukkan adanya banyak _outlier_ pada hampir semua fitur, yang mengindikasikan perlunya penanganan _outlier_ pada tahap persiapan data.
- **Analisis Korelasi:** Heatmap korelasi menunjukkan bahwa `alcohol` memiliki korelasi positif terkuat dengan `quality` (0.48), sementara `volatile acidity` memiliki korelasi negatif terkuat (-0.39).

## Data Preparation

Teknik persiapan data yang dilakukan secara berurutan sesuai dengan notebook adalah:

1. **Transformasi Variabel Target:** Variabel `quality` (skala 3-8) diubah menjadi variabel biner `quality_category`. Anggur dengan `quality` > 5 dikategorikan sebagai 'Baik' (1) dan `quality` <= 5 sebagai 'Tidak Baik' (0).
2. **Penanganan Data Duplikat:** Sebanyak 240 baris data yang terduplikasi dihapus dari dataset untuk menghindari bias pada model. Dataset berkurang menjadi 1359 sampel.
3. **Penanganan Outlier:** _Outlier_ pada semua fitur input ditangani menggunakan metode IQR. Nilai yang berada di luar batas (Q1 - 1.5*IQR dan Q3 + 1.5*IQR) digantikan dengan nilai batas tersebut (_capping/winsorization_).
4. **Pemisahan Data:** Dataset dibagi menjadi data latih (1087 sampel) dan data uji (272 sampel) dengan proporsi 80:20. Parameter `stratify=y` digunakan untuk memastikan distribusi kelas target tetap proporsional di kedua set.
5. **Penskalaan Fitur (Standardization):** Semua fitur input distandarisasi menggunakan `StandardScaler`. Skaler ini dilatih pada data latih dan kemudian digunakan untuk mentransformasi data latih dan data uji.

- **Proses dan Alasan Data Preparation:**

Proses persiapan data dimulai dengan mengubah variabel target quality menjadi format biner (quality_category) untuk menyederhanakan masalah menjadi klasifikasi. Setelah inspeksi awal, ditemukan 240 baris data duplikat yang kemudian dihapus untuk memastikan setiap sampel bersifat unik. Selanjutnya, dilakukan penanganan outlier menggunakan metode IQR; nilai-nilai ekstrem pada setiap fitur digantikan dengan nilai ambang batasnya (capping) untuk mengurangi distorsi pada model. Setelah data bersih, dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan metode pemisahan terstratifikasi untuk menjaga proporsi kelas. Terakhir, semua fitur input pada data latih dan uji distandarisasi menggunakan StandardScaler. Proses fitting skaler ini hanya dilakukan pada data latih untuk mencegah kebocoran informasi dari data uji, yang kemudian digunakan untuk mentransformasi kedua set data tersebut.

Setiap tahapan persiapan data memiliki tujuan spesifik.

- **Transformasi Target:** Diperlukan untuk mengubah masalah menjadi format klasifikasi biner yang sesuai dengan tujuan proyek.
- **Penanganan Duplikat:** Krusial untuk memastikan setiap sampel data unik dan mencegah model menjadi bias terhadap sampel yang berulang.
- **Penanganan Outlier:** Dilakukan untuk mengurangi pengaruh nilai-nilai ekstrem yang dapat mengganggu proses pembelajaran model dan metrik evaluasi. Metode _capping_ dipilih agar tidak kehilangan data.
- **Pemisahan Data:** Praktik standar untuk mengevaluasi kemampuan generalisasi model pada data yang belum pernah dilihat sebelumnya dan mencegah _overfitting_.
- **Penskalaan Fitur:** Penting karena algoritma seperti Logistic Regression dan SVM sensitif terhadap skala fitur. Standarisasi memastikan semua fitur memiliki kontribusi yang setara pada proses pembelajaran.

## Modeling

Tahapan ini membahas model _machine learning_ yang digunakan.

1. **Pemilihan Algoritma:** Tiga model klasifikasi digunakan:
   - Logistic Regression (sebagai _baseline_)
   - Random Forest Classifier
   - Support Vector Machine (SVM)
2. **Pelatihan Model:** Ketiga model dilatih menggunakan data latih yang telah diproses dengan parameter _default_ untuk perbandingan awal.
3. **Pemilihan Model Terbaik & Hyperparameter Tuning:**
   - Berdasarkan evaluasi awal pada data uji, **SVM** menunjukkan F1-Score tertinggi (0.7722) dan dipilih sebagai model terbaik untuk dioptimalkan.
   - Model SVM kemudian dioptimalkan menggunakan `GridSearchCV` untuk mencari kombinasi parameter terbaik dari `C`, `kernel`, dan `gamma`.

- **Kelebihan dan Kekurangan Algoritma:**

  - **Logistic Regression:** - _Kelebihan:_ Cepat, mudah diinterpretasikan, dan baik sebagai _baseline_. - _Kekurangan:_ Mengasumsikan hubungan linear, kurang efektif pada masalah yang kompleks.
  - **Random Forest Classifier:** - _Kelebihan:_ Efektif pada masalah non-linear, tahan terhadap _overfitting_ (dibanding _decision tree_ tunggal), dan dapat memberikan fitur penting. - _Kekurangan:_ Lebih lambat, kurang interpretatif (_black box_). Dalam proyek ini, model ini menunjukkan _overfitting_ yang signifikan (akurasi latih 1.0 vs uji 0.7610).
  - **Support Vector Machine (SVM):** - _Kelebihan:_ Efektif di ruang berdimensi tinggi, fleksibel dengan berbagai _kernel_, dan memiliki performa yang kuat. - _Kekurangan:_ Kurang efisien pada dataset yang sangat besar, sensitif terhadap pemilihan _kernel_ dan parameter.

  - **Pemilihan Model Terbaik dan Proses Improvement:**
    - Model **SVM (dengan parameter _default_)** dipilih sebagai solusi terbaik karena memberikan **F1-Score tertinggi (0.7722)** pada data uji, mengungguli Logistic Regression (0.7245) dan Random Forest (0.7612).
    - Proses _improvement_ dilakukan dengan `GridSearchCV` pada model SVM. Hasilnya, parameter terbaik yang ditemukan adalah `{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}`. Namun, model yang telah di-_tuning_ ini justru menunjukkan performa sedikit lebih rendah (F1-Score 0.7539) pada data uji dibandingkan model SVM _default_.
    - Oleh karena itu, **model akhir yang dipilih sebagai solusi terbaik adalah SVM dengan parameter _default_**, karena menunjukkan kemampuan generalisasi yang paling unggul pada data yang belum pernah dilihat.

## Evaluation

Pada bagian ini, dilakukan evaluasi terhadap model-model yang telah dilatih untuk mengukur kinerjanya dalam memprediksi kualitas anggur.

### Metrik Evaluasi yang Digunakan

Berikut adalah metrik-metrik yang digunakan untuk mengevaluasi performa model:

- **Confusion Matrix:** Sebuah tabel untuk visualisasi performa yang menunjukkan jumlah prediksi benar dan salah (True Positive, True Negative, False Positive, dan False Negative).
- **Akurasi (Accuracy):** Persentase total prediksi yang benar. Memberikan gambaran umum kinerja model.
- **Presisi (Precision):** Rasio prediksi positif yang benar terhadap total prediksi positif. Penting untuk meminimalkan kesalahan pelabelan positif (False Positive).
- **Recall (Sensitivity):** Rasio prediksi positif yang benar terhadap total data yang sebenarnya positif. Penting untuk meminimalkan sampel positif yang terlewat (False Negative).
- **F1-Score:** Rata-rata harmonik dari Presisi dan Recall. Menjadi metrik acuan utama karena memberikan keseimbangan antara Presisi dan Recall, terutama jika ada potensi ketidakseimbangan kelas.
- **Classification Report:** Laporan teks yang merangkum nilai Presisi, Recall, dan F1-Score untuk setiap kelas secara individual.
- **Permutation Importance:** Teknik untuk mengukur seberapa besar pengaruh sebuah fitur terhadap performa model, digunakan untuk interpretasi model terbaik.

### Hasil Proyek Berdasarkan Metrik Evaluasi

Tabel berikut merangkum hasil evaluasi dari semua model yang diuji pada _test set_:

| Model                   | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :---------------------- | :------------- | :------------ | :------------- | :---------- | :------------ |
| Logistic Regression     | 0.7470         | 0.7243        | 0.7257         | 0.7243      | 0.7245        |
| Random Forest (Default) | 1.0000         | 0.7610        | 0.7624         | 0.7610      | 0.7612        |
| **SVM (Default)**       | **0.8050**     | **0.7721**    | **0.7745**     | **0.7721**  | **0.7722**    |
| SVM (Tuned)             | 0.7893         | 0.7537        | 0.7546         | 0.7537      | 0.7539        |

(Semua metrik Precision, Recall, dan F1-Score adalah 'weighted average')

- **Model Terbaik:** **SVM (Default)** terpilih sebagai model terbaik dengan **F1-Score 0.7722**. Model ini memberikan keseimbangan terbaik antara Presisi dan Recall, serta menunjukkan generalisasi yang baik karena perbedaan kecil antara akurasi pada data latih dan data uji.
- **Analisis Fitur Penting:** Menggunakan metode **Permutation Importance** pada model SVM terbaik, ditemukan bahwa fitur-fitur yang paling berpengaruh dalam menentukan kualitas anggur adalah:
  1. `alcohol`
  2. `sulphates`
  3. `volatile acidity` 4.`total sulfur dioxide`
     Hasil ini menjawab salah satu tujuan bisnis utama, yaitu mengidentifikasi faktor-faktor kunci penentu kualitas anggur.

### Penjelasan Formula dan Cara Kerja Metrik

#### **1. Confusion Matrix**

- **Struktur:** Tabel N x N yang memvisualisasikan performa klasifikasi. Untuk klasifikasi biner, komponennya adalah:
  - **True Positive (TP):** Data positif yang diprediksi benar.
  - **True Negative (TN):** Data negatif yang diprediksi benar.
  - **False Positive (FP):** Data negatif yang salah diprediksi positif (Error Tipe I).
  - **False Negative (FN):** Data positif yang salah diprediksi negatif (Error Tipe II).
- **Konteks Proyek:** Membantu melihat secara detail di mana model membuat kesalahan, misalnya berapa banyak anggur berkualitas buruk yang salah dilabeli sebagai berkualitas baik (FP).

#### **2. Akurasi (Accuracy)**

- **Formula:** $$\text{Akurasi} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
- **Konteks Proyek:** Memberikan gambaran umum seberapa sering model berhasil menebak kualitas anggur dengan tepat secara keseluruhan.

#### **3. Presisi (Precision)**

- **Formula:** $$\text{Presisi} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Konteks Proyek:** Menjawab pertanyaan, "Dari semua anggur yang diprediksi berkualitas baik, berapa persen yang _benar-benar_ baik?". Penting untuk menghindari kekecewaan konsumen akibat pelabelan yang salah.

#### **4. Recall (Sensitivity)**

- **Formula:** $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Konteks Proyek:** Menjawab pertanyaan, "Dari semua anggur yang _sebenarnya_ berkualitas baik, berapa persen yang berhasil ditemukan oleh model?". Penting agar tidak ada produk unggulan yang terlewatkan.

#### **5. F1-Score**

- **Formula:** $$\text{F1-Score} = 2 \times \frac{\text{Presisi} \times \text{Recall}}{\text{Presisi} + \text{Recall}}$$
- **Konteks Proyek:** Menjadi metrik utama karena memberikan skor tunggal yang menyeimbangkan antara pentingnya menghindari kesalahan pelabelan (Presisi) dan pentingnya menemukan semua produk bagus (Recall).

#### **6. Classification Report**

- **Struktur:** Laporan teks yang merangkum nilai **Presisi, Recall, dan F1-Score** untuk setiap kelas secara terpisah.
- **Konteks Proyek:** Krusial untuk menganalisis apakah model memiliki kecenderungan lebih baik dalam memprediksi satu kelas dibandingkan kelas lainnya.

#### **7. Permutation Importance**

- **Struktur:** `Importance(F) = Score_baseline - Score_permuted(F)`
- **Konteks Proyek:** Digunakan setelah model terbaik dipilih untuk menjawab pertanyaan bisnis mengenai faktor kimia apa saja yang paling berpengaruh dalam menentukan kualitas anggur.
