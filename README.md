# Laporan Proyek Machine Learning - Salman Al Farizi Harahap

## Domain Proyek

### Latar Belakang
Perubahan iklim dan variabilitas cuaca yang semakin tidak menentu menuntut adanya metode peramalan cuaca yang lebih akurat. Peramalan tren cuaca dapat membantu berbagai sektor, termasuk pertanian, transportasi, dan kebijakan lingkungan, dalam mengambil keputusan yang lebih tepat. Dengan kemajuan teknologi machine learning dan deep learning, peramalan cuaca dapat dilakukan secara lebih efektif menggunakan data historis.

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
Dataset yang digunakan berasal dari [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/data) di Kaggle, yang mencakup berbagai parameter cuaca dari berbagai lokasi di dunia. Dataset ini memiliki jumlah entri yang cukup besar untuk memungkinkan analisis tren cuaca secara komprehensif. Meskipun tidak terdapat nilai yang hilang dalam dataset, beberapa variabel memerlukan transformasi untuk memastikan bahwa model machine learning dapat memprosesnya secara efektif.

### Variabel-variabel pada Global Weather Repository adalah sebagai berikut:
- country: Negara tempat data cuaca berada
- location_name: Nama lokasi (kota)
- latitude: Koordinat lintang lokasi
- longitude: Koordinat bujur lokasi
- timezone: Zona waktu lokasi
- last_updated_epoch: Stempel waktu Unix dari pembaruan data terakhir
- last_updated: Waktu lokal dari pembaruan data terakhir
- temperature_celsius: Suhu dalam derajat Celsius
- temperature_fahrenheit: Suhu dalam derajat Fahrenheit
- condition_text: Deskripsi kondisi cuaca
- wind_mph: Kecepatan angin dalam mil per jam
- wind_kph: Kecepatan angin dalam kilometer per jam
- wind_degree: Arah angin dalam derajat
- wind_direction: Arah angin sebagai kompas 16 titik
- pressure_mb: Tekanan dalam milibar
- pressure_in: Tekanan dalam inci
- precip_mm: Jumlah curah hujan dalam milimeter
- precip_in: Jumlah curah hujan dalam inci
- humidity: Kelembaban sebagai persentase
- cloud: Tutupan awan sebagai persentase
- feels_like_celsius: Suhu yang terasa dalam Celsius
- feels_like_fahrenheit: Suhu yang terasa dalam Fahrenheit
- visibility_km: Visibilitas dalam kilometer
- visibility_miles: Visibilitas dalam mil
- uv_index: Indeks UV
- gust_mph: Hembusan angin dalam mil per jam
- gust_kph: Hembusan angin dalam kilometer per jam
- air_quality_Carbon_Monoxide: Pengukuran kualitas udara: Karbon Monoksida
- air_quality_Ozone: Pengukuran kualitas udara: Ozon
- air_quality_Nitrogen_dioxide: Pengukuran kualitas udara: Nitrogen Dioksida
- air_quality_Sulphur_dioxide: Pengukuran kualitas udara: Sulfur Dioksida
- air_quality_PM2.5: Pengukuran kualitas udara: PM2.5
- air_quality_PM10: Pengukuran kualitas udara: PM10
- air_quality_us-epa-index: Pengukuran kualitas udara: Indeks EPA AS
- air_quality_gb-defra-index: Pengukuran kualitas udara: Indeks DEFRA GB
- sunrise: Waktu matahari terbit setempat
- sunset: Waktu setempat saat matahari terbenam
- moonrise: Waktu setempat saat bulan terbit
- moonset: Waktu setempat saat bulan terbenam
- moon_phase: Fase bulan saat ini
- moon_illumination: Persentase iluminasi bulan

### Explorasi Data

* Data divisualisasikan menggunakan histogram dan heatmap untuk memahami distribusi dan korelasi antar variabel.

![Heat Map](https://github.com/salmanafh/weather-trend-forcasting/blob/d71be5c35a6583b092a377606b1695b20a6a88c7/corr%20map.png)

* Selain itu, analisis tren dilakukan untuk memahami pola perubahan cuaca dari waktu ke waktu. Grafik tren kecepatan angin menunjukkan adanya pola musiman yang signifikan,

![Wind Speed Trend](https://github.com/salmanafh/weather-trend-forcasting/blob/main/wind%20speed%20trend.png)

* Peningkatan suhu pada bulan-bulan tertentu dan penurunan pada bulan lainnya

![Temperature Trend](https://github.com/salmanafh/weather-trend-forcasting/blob/main/temperature%20trend.png)

* Curah hujan juga menunjukkan pola fluktuatif yang berkaitan dengan perubahan musim. Informasi ini membantu dalam mengidentifikasi pola utama yang dapat digunakan untuk meningkatkan akurasi model prediksi cuaca.



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

