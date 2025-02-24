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

* XGBoost: Algoritma probabilistik yang sederhana namun efektif dalam klasifikasi data cuaca.

* Random Forest: Algoritma ensemble yang mampu menangani data kompleks dan meningkatkan akurasi.

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

![corr map](https://github.com/user-attachments/assets/8f0c50c3-25f0-4c41-9719-c4fdb41fde70)

* Selain itu, analisis tren dilakukan untuk memahami pola perubahan cuaca dari waktu ke waktu. Grafik tren kecepatan angin menunjukkan adanya pola musiman yang signifikan,

![wind speed trend](https://github.com/user-attachments/assets/5d4c60c1-3bec-4c0f-990c-c6f41766292b)

* Peningkatan suhu pada bulan-bulan tertentu dan penurunan pada bulan lainnya

![temperature trend](https://github.com/user-attachments/assets/510b1db9-1e11-43ce-984d-78da14785f41)

* Curah hujan juga menunjukkan pola fluktuatif yang berkaitan dengan perubahan musim. Informasi ini membantu dalam mengidentifikasi pola utama yang dapat digunakan untuk meningkatkan akurasi model prediksi cuaca.

![humidity trend](https://github.com/user-attachments/assets/839a3bfe-adb7-4b01-8bab-043fee8afac3)

## Data Preparation

* One-Hot Encoding: Mengonversi data kategori menjadi bentuk numerik untuk memastikan bahwa model dapat memahami informasi dari variabel kategori dengan benar. Teknik ini digunakan agar data dapat digunakan oleh algoritma machine learning yang sebagian besar hanya dapat bekerja dengan nilai numerik.

~~~
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[categorical_columns])
encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

df = pd.concat([df, encoded_df], axis=1)
df = df.drop(categorical_columns, axis=1)
~~~

* Feature Scaling: Normalisasi atau standarisasi fitur dilakukan untuk memastikan bahwa semua variabel berada dalam skala yang sama. Hal ini sangat penting karena model machine learning, terutama yang berbasis gradien seperti XGBoost atau ANN, lebih sensitif terhadap perbedaan skala antar fitur.

~~~
df[numerical_columns] = preprocessing.normalize(df[numerical_columns])
~~~

* Splitting Data: Data dibagi menjadi set pelatihan dan pengujian untuk menghindari overfitting serta mengevaluasi kinerja model pada data yang belum pernah dilihat sebelumnya. Proporsi pembagian data yang umum digunakan adalah 80% untuk pelatihan dan 20% untuk pengujian.

~~~
X = df.drop(["condition_text", "last_updated"], axis=1)
y = encoded_label.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
~~~

## Modeling

### Model 1: XGBoost
* Keunggulan: Cepat, efisien, mampu menangani dataset besar, memiliki mekanisme regularisasi untuk mengatasi overfitting, dan bekerja baik dengan data yang tidak seimbang.
* Kelemahan: Membutuhkan lebih banyak sumber daya komputasi, rentan terhadap overfitting tanpa tuning yang tepat, dan memiliki waktu pelatihan yang lebih lama dibandingkan model sederhana..

### Model 2: Random Forest
* Keunggulan: Mampu menangani fitur yang kompleks dan tidak linear.
* Kelemahan: Rentan terhadap overfitting jika tidak ditangani dengan baik.

### Model 3: Deep Learning (ANN)
* Keunggulan: Bisa Menangkap pola non-linear yang kompleks.
* Kelemahan: Membutuhkan lebih banyak data dan waktu pelatihan yang lebih lama.

## Evaluation

Metric Berikut menjelaskan model dengan performa terbaik
### Accuracy

Mengukur persentase prediksi yang benar terhadap total prediksi.

$$ Accuracy = TP + TN / N $$

![Model Accuracy](https://github.com/salmanafh/weather-trend-forcasting/blob/205b0f0a9fb2341318496833e96965a83a59cfce/validation%20accuracy.png)

Dari grafik, model Neural Network dan XGB memiliki akurasi lebih tinggi dibandingkan Random Forest.

### Recall

Recall mengukur proporsi kasus positif yang berhasil diidentifikasi dengan benar.

$$ Recall = TP / (TP + FN) $$

![Model Recall](https://github.com/salmanafh/weather-trend-forcasting/blob/main/Recall.png)

Model Neural Network dan XGB memiliki recall yang lebih tinggi dibandingkan Random Forest, menunjukkan bahwa keduanya lebih baik dalam mengenali kasus positif.

### Precision

Precision mengukur proporsi prediksi positif yang benar terhadap total prediksi positif.

$$ Precision = TP / (TP + FP) $$

![Model Precision](https://github.com/salmanafh/weather-trend-forcasting/blob/main/Precision.png)

Model XGB dan Random Forest memiliki precision yang lebih tinggi dibandingkan Neural Network.

### F1 Score

F1 score adalah rata-rata harmonik dari precision dan recall, digunakan untuk menyeimbangkan false positives dan false negatives.

$$ F1 Score = 2 * (Precision * Recall) / (Precision + Recall) $$

![Model Recall](https://github.com/salmanafh/weather-trend-forcasting/blob/main/F1%20Score.png)

Model XGB dan Neural Network memiliki F1-score lebih tinggi dibandingkan Random Forest, menunjukkan bahwa keduanya lebih seimbang dalam menangani positif dan negatif.
