# Submission 1: Nama Proyek Anda
Nama:

Username dicoding: Yusril Hasan

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [political-bias](https://www.kaggle.com/datasets/mayobanexsantana/political-bias) |
| Masalah | Masalah yang diambil adalah banyak berita yang seharusnya narasi yang relevan dan berdasakran fakta, tetapi dibumbui oleh bias. Terkhusus untuk topik terkait politic. Pada kasus ini, saya mengambil data yang berupa kumpulan berita yang mengandung antara 5 jenis ini political bias dalam lingkup Amerika Serikut: {"left": 1, "lean left": 2, "center": 3, "lean right": 4, "right": 5} |
| Solusi machine learning | Saya menggunakan metode deep learning pada case multiclass classififcation untuk memprediksi text berita mana yang mengandung political bias |
| Metode pengolahan | Untuk metode pengolahan nya sendiri saya menggunakan manual clining, sebagaimana yang terlihat pada notebook.ipynb section Manual Clining yang dimulai dari: 1. label encoders sesuai yang terdapat pada mapping.json, 2.  membersihkan additional characters, 3. Drop NA values, 4. Membuang kolom selain bias (label) dan text (features)  |
| Arsitektur model | Model yang saya gunakan berupa 7 layers, yang terdiri dari input, embedding, GlobalAveragePooling1D, Dense, dan output layers. Disini saya juga menggunakan activation function berupa softmax dengan loss function categorical_crossentropy |
| Metrik evaluasi | Karena ini merupakan kasus multiclass clasification NLP, jadi saya menggunakan metrics: CategoricalAccuracy dan AUC |
| Performa model | Dari CategoricalAccuracy: 0.5273, sedangkan AUC: 0.7801. Bisa dilihat kalau dari segi CategoricalAccuracy masih kurang bagus sedangkan untuk AUC sudah cukup bagus. Hal ini menandakan perlunya thresholding pada output result agar nilai CategoricalAccuracy bisa naik. Selain itu, bisa dibilang kalau model ini juga sedikit underfitting dikarenakan kurangnya data.|
