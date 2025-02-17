# Laporan Proyek Machine Learning - Salman Al Farizi Harahap

## Domain Proyek

### Latar Belakang
Perubahan iklim dan variabilitas cuaca yang semakin tidak menentu menuntut adanya metode peramalan cuaca yang lebih akurat. Peramalan tren cuaca dapat membantu berbagai sektor, termasuk pertanian, transportasi, dan kebijakan lingkungan, dalam mengambil keputusan yang lebih tepat. Dengan kemajuan teknologi machine learning dan deep learning, peramalan cuaca dapat dilakukan secara lebih efektif menggunakan data historis.

**Rubrik/Kriteria Tambahan (Opsional)**:
### Urgensi
Cuaca yang tidak dapat diprediksi dengan baik dapat menyebabkan berbagai permasalahan, seperti gagal panen, gangguan transportasi, dan dampak buruk pada kesehatan masyarakat. Oleh karena itu, diperlukan model yang dapat mengidentifikasi pola cuaca berdasarkan data historis dan memberikan prediksi yang akurat.

### Referensi dan Riset terkait
Beberapa penelitian telah membuktikan bahwa machine learning dan deep learning dapat meningkatkan akurasi prediksi cuaca. Misalnya:

* ["Weather Forecasting Using Machine Learning Techniques" (Sharma et al., 2020)](https://ieeexplore.ieee.org/abstract/document/10434670)

* ["A Comparative Study of Machine Learning Models for Weather Prediction" (Singh et al., 2021)](https://www.researchgate.net/publication/355783139_A_comparative_study_of_extensive_machine_learning_models_for_predicting_long-term_monthly_rainfall_with_an_ensemble_of_climatic_and_meteorological_predictors)

## Business Understanding

### Problem Statements

* Bagaimana cara mengembangkan model machine learning yang dapat memprediksi tren cuaca secara akurat?

* Algoritma mana yang memberikan performa terbaik dalam peramalan tren cuaca?

### Goals

* Mengembangkan model machine learning yang mampu memprediksi tren cuaca dengan akurasi tinggi.

* Mengevaluasi performa model menggunakan metrik evaluasi yang sesuai.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
### Solusi yang Diusulkan
Dua pendekatan utama yang digunakan:

* Naive Bayes: Algoritma probabilistik yang sederhana namun efektif dalam klasifikasi data cuaca.

* Random Forest & XGBoost: Algoritma ensemble yang mampu menangani data kompleks dan meningkatkan akurasi.

* Deep Learning (Artificial Neural Networks): Model berbasis jaringan saraf untuk menangkap pola non-linear dalam data cuaca.

### Metric Evaluasi

* Accuracy: Mengukur persentase prediksi yang benar.

* Precision, Recall, dan F1-score: Digunakan untuk mengevaluasi performa model dalam menangani kelas yang tidak seimbang.

* Confusion Matrix: Menganalisis distribusi prediksi yang benar dan salah.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

