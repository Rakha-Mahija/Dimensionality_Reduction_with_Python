# Dimensionality Reduction pada Dataset Wine Quality: Studi Perbandingan PCA dan t-SNE

## Deskripsi Kasus

Setiap botol wine merah menyimpan jejak kimiawi yang kompleks: kadar asam, gula sisa, klorida, sulfur dioksida, densitas, pH, sulfat, hingga kandungan alkohol — total ada 11 karakteristik numerik yang bersama-sama menentukan bagaimana kualitas sebuah wine dinilai. Bagi manusia, sebelas dimensi ini adalah ruang yang mustahil untuk "dilihat" secara langsung. Kita bisa membuat scatter plot dua fitur sekaligus, tapi begitu jumlah fitur melebihi tiga, intuisi visual kita berhenti bekerja.

Di sinilah masalah utama proyek ini bermula. Ketika seorang analis ingin memahami apakah wine dengan kualitas tinggi dan rendah membentuk kelompok yang berbeda secara kimiawi, ia tidak bisa begitu saja memplot 11 sumbu dalam satu grafik. Ia butuh cara untuk "meringkas" informasi dari 11 dimensi tersebut ke dalam ruang yang bisa dilihat mata manusia — idealnya 2 atau 3 dimensi — tanpa kehilangan terlalu banyak makna dari data aslinya.

Kebutuhan inilah yang dijawab oleh teknik **dimensionality reduction**. Proyek ini berfokus pada kasus **visualisasi data berdimensi tinggi**: mereduksi 11 fitur kimiawi wine menjadi representasi 2D, dengan tujuan mengamati apakah pola persebaran dan kecenderungan pengelompokan berdasarkan kualitas wine dapat terlihat secara visual. Dua pendekatan dipakai dan dibandingkan — PCA yang bekerja secara linear, dan t-SNE yang menangkap struktur non-linear — untuk melihat pendekatan mana yang memberikan gambaran paling informatif.

## Dataset yang Digunakan

Dataset yang dipakai adalah **Wine Quality (Red Wine variant)**, dataset publik dari UCI Machine Learning Repository yang banyak digunakan sebagai studi kasus dalam pembelajaran machine learning.

| Atribut | Keterangan |
|---|---|
| Nama dataset | Wine Quality – Red Wine |
| Jumlah baris | 1.599 sampel wine |
| Jumlah fitur | 11 fitur numerik (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) |
| Target/label | `quality` (skor kualitas wine, digunakan hanya sebagai pewarnaan visualisasi, bukan sebagai target modeling) |
| Sumber | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) |

Sebelum diproses, data dicek kelengkapannya (tidak ditemukan missing value) dan seluruh fitur diskalakan menggunakan `StandardScaler` agar memiliki rata-rata 0 dan varians 1 — langkah wajib karena PCA dan t-SNE sama-sama sensitif terhadap perbedaan skala antar fitur (misalnya `total sulfur dioxide` yang bernilai puluhan, berbanding `density` yang bernilai mendekati 1).

## Ringkasan Metode

Alur kerja proyek ini mengikuti pipeline standar untuk eksplorasi data berdimensi tinggi:

1. **Pra-pemrosesan** — Kolom target `quality` dipisahkan dari 11 fitur kimiawi, kemudian seluruh fitur distandarisasi dengan `StandardScaler`.
2. **Principal Component Analysis (PCA)** — Data 11 dimensi direduksi menjadi 2 komponen utama (PC1 dan PC2). PCA bekerja dengan mencari arah-arah baru (kombinasi linear dari fitur asli) yang memaksimalkan varians data, sehingga komponen pertama menangkap sebanyak mungkin informasi, disusul komponen kedua, dan seterusnya. Selain proyeksi 2D, dihitung pula *explained variance ratio* untuk mengukur seberapa banyak informasi asli yang berhasil dipertahankan.
3. **t-Distributed Stochastic Neighbor Embedding (t-SNE)** — Data yang sama direduksi ke 2 dimensi menggunakan t-SNE (`perplexity=30`, `learning_rate='auto'`, `init='pca'`). Berbeda dari PCA, t-SNE tidak mencoba mempertahankan varians global, melainkan berfokus menjaga kedekatan (jarak lokal) antar titik data yang mirip, sehingga titik-titik yang berdekatan di ruang asli cenderung tetap berdekatan di ruang 2D hasil reduksi.
4. **Visualisasi** — Kedua hasil reduksi diplot sebagai scatter plot 2D dengan pewarnaan berdasarkan skor `quality`, untuk mengamati apakah ada kecenderungan pengelompokan visual.
5. **Perbandingan** — Karakteristik kedua metode (linear vs non-linear, kecepatan komputasi, sifat deterministik, dan tujuan penggunaan) dibandingkan untuk menarik kesimpulan metode mana yang paling sesuai untuk kasus visualisasi ini.

## Analisis Hasil

### Apa yang ditunjukkan PCA

Ketika 11 fitur kimiawi wine diproyeksikan ke dua komponen utama, PCA berhasil mempertahankan sekitar **45,68% dari total varians data** — angka yang cukup moderat, menandakan bahwa keragaman wine tidak sepenuhnya bisa dijelaskan hanya dari dua arah linear saja. Ini masuk akal secara kimiawi: 11 senyawa dalam wine tidak selalu berkorelasi linear satu sama lain, sehingga ruang informasinya memang sulit dipadatkan tanpa kehilangan sebagian besar variasi.

Secara visual, hasil PCA memperlihatkan sebagian besar titik data menumpuk di satu area besar berbentuk awan tanpa batas yang jelas. Wine dengan kualitas tinggi dan rendah memang menunjukkan sedikit kecenderungan posisi yang berbeda di sepanjang sumbu PC1, tetapi batas antar kelas kualitas tidak tegas — mereka saling tumpang tindih. Ini adalah temuan yang wajar untuk PCA: karena metode ini hanya melakukan rotasi dan proyeksi linear, ia unggul dalam merangkum arah variasi terbesar dalam data, tetapi kurang peka terhadap struktur kelompok yang bersifat non-linear.

### Apa yang ditunjukkan t-SNE

Hasil berbeda muncul ketika data yang sama diproses dengan t-SNE. Alih-alih satu awan besar yang menumpuk, t-SNE menghasilkan sebaran yang lebih terpecah menjadi kelompok-kelompok lokal yang lebih terdefinisi. Titik-titik yang secara kimiawi mirip cenderung ditempatkan berdekatan, membentuk gugusan-gugusan kecil di seluruh bidang 2D.

Meskipun label kualitas masih terlihat bercampur di dalam gugusan-gugusan tersebut — artinya kualitas wine bukan satu-satunya faktor yang membentuk kemiripan kimiawi — pola pengelompokan yang muncul mengindikasikan bahwa hubungan antar fitur kimiawi wine memang bersifat non-linear. Wine-wine dengan profil kimia serupa (kombinasi asam, gula, alkohol, dsb.) cenderung mengelompok bersama terlepas dari skor kualitasnya, yang menunjukkan bahwa faktor kimiawi punya struktur tersembunyi yang tidak tertangkap oleh proyeksi linear semata.

### Membandingkan Keduanya

Perbedaan mendasar antara PCA dan t-SNE pada dasarnya adalah perbedaan filosofi: PCA menjawab pertanyaan "arah mana yang menyimpan variasi terbesar dari keseluruhan data?", sedangkan t-SNE menjawab pertanyaan "titik mana yang sebenarnya bertetangga dekat satu sama lain?". Konsekuensinya terlihat jelas pada karakteristik berikut:

| Aspek | PCA | t-SNE |
|---|---|---|
| Sifat transformasi | Linear | Non-linear |
| Kecepatan komputasi | Sangat cepat | Relatif lambat |
| Fokus utama | Mempertahankan varians global | Mempertahankan struktur/jarak lokal |
| Konsistensi hasil | Deterministik (hasil sama tiap dijalankan) | Stokastik (hasil bisa sedikit berbeda tiap run) |
| Interpretasi jarak | Jarak & skala relatif punya makna | Jarak antar cluster tidak selalu bermakna kuantitatif |

Karena sifat-sifat inilah, PCA cenderung memberi gambaran "peta besar" dari data — ke arah mana data paling bervariasi — namun kurang tajam dalam memisahkan kelompok yang tumpang tindih secara linear. Sebaliknya, t-SNE mengorbankan kecepatan dan interpretasi jarak global demi mendapatkan gambaran kelompok lokal yang lebih tajam, meski hasilnya bisa sedikit berubah setiap kali dijalankan ulang.

### Metode Mana yang Lebih Sesuai?

Untuk kasus **visualisasi data berdimensi tinggi** seperti pada proyek ini — di mana tujuan utamanya adalah mengamati pola persebaran dan kemungkinan pengelompokan wine berdasarkan karakteristik kimiawinya — **t-SNE adalah pilihan yang lebih sesuai**. Alasannya sederhana: tujuan proyek ini bukan mengukur seberapa besar variasi yang bisa dijelaskan secara linear, melainkan untuk *melihat* apakah ada pola pengelompokan yang tersembunyi di dalam data. t-SNE dirancang khusus untuk tujuan ini, karena ia menjaga hubungan ketetanggaan lokal sehingga kelompok-kelompok data yang secara kimiawi mirip akan tampak mengelompok secara visual — sesuatu yang tidak bisa dilakukan PCA dengan baik ketika hubungan antar fitur bersifat non-linear.

Meski demikian, PCA tetap memegang peran yang tidak tergantikan di luar konteks visualisasi murni. Sifatnya yang cepat, deterministik, dan memberikan ukuran kuantitatif (*explained variance*) menjadikannya pilihan yang lebih tepat untuk tahap awal eksplorasi data, kompresi dimensi sebelum proses modeling machine learning, atau ketika dibutuhkan hasil yang konsisten dan dapat direproduksi.

Dengan kata lain, kedua metode ini tidak saling menggantikan, melainkan saling melengkapi: **PCA cocok digunakan lebih dulu** untuk memahami struktur variasi data secara cepat dan efisien, sementara **t-SNE digunakan kemudian** — khususnya ketika tujuan akhirnya adalah menghasilkan visualisasi yang tajam untuk mengungkap pola pengelompokan yang tidak terlihat oleh metode linear seperti PCA.
