# Book Recommendation System

Aplikasi rekomendasi buku menggunakan **Item-Based Collaborative Filtering** dengan **Cosine Similarity** dan **K-Nearest Neighbors (KNN)**.

## Dataset

Aplikasi menggunakan dataset Book-Crossing:
- `BX-Books.csv`
- `BX-Book-Ratings.csv`
- `BX-Users.csv` (opsional; tidak digunakan langsung dalam pembentukan model)

Unggah file CSV melalui aplikasi Streamlit. File CSV harus menggunakan pemisah titik koma (`;`) dan encoding `latin-1`.

## Metode

1. Menggunakan rating eksplisit (`Book-Rating > 0`).
2. Memilih pengguna dengan minimal 5 rating.
3. Memilih buku dengan minimal 20 rating.
4. Membentuk matriks User-Item.
5. Mengubah matriks menjadi Item-User.
6. Melatih `NearestNeighbors` dengan metrik cosine.
7. Menghitung skor rekomendasi berdasarkan rata-rata rating berbobot kemiripan.

## Instalasi dan Menjalankan Aplikasi

Pastikan Python sudah terpasang. Jalankan perintah berikut pada terminal di folder proyek:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Kemudian buka alamat lokal yang ditampilkan di terminal oleh Streamlit.
