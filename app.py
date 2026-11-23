import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System")
st.write(
    "Sistem rekomendasi buku menggunakan **Item-Based Collaborative Filtering** "
    "dengan kemiripan **Cosine Similarity** dan algoritma **K-Nearest Neighbors (KNN)**."
)

st.markdown("""
Aplikasi ini menggunakan rating eksplisit (rating lebih dari 0). Pengguna yang
memiliki minimal 5 rating dan buku yang memiliki minimal 20 rating digunakan
dalam pembentukan model.
""")

@st.cache_data(show_spinner=False)
def read_csv_file(file, required_columns):
    """Membaca CSV Book-Crossing dengan pemisah titik koma dan encoding latin-1."""
    df = pd.read_csv(
        file,
        sep=";",
        encoding="latin-1",
        dtype={"ISBN": str}
    )
    df.columns = df.columns.str.strip()

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}"
        )
    return df


def clean_isbn(value):
    """Menyeragamkan format ISBN agar dapat dicocokkan antar dataset."""
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    # Mengembalikan nol di depan untuk ISBN numerik yang terbaca lebih pendek.
    if value.isdigit() and len(value) < 10:
        value = value.zfill(10)

    return value


@st.cache_resource(show_spinner="Sedang membangun model rekomendasi...")
def build_recommendation_model(books_file, ratings_file):
    """Preprocessing data, membentuk matriks User-Item, dan melatih model KNN."""
    books = read_csv_file(
        books_file,
        ["ISBN", "Book-Title", "Book-Author"]
    )
    ratings = read_csv_file(
        ratings_file,
        ["User-ID", "ISBN", "Book-Rating"]
    )

    # Preprocessing data buku
    books = books.copy()
    ratings = ratings.copy()

    books["ISBN"] = books["ISBN"].apply(clean_isbn)
    ratings["ISBN"] = ratings["ISBN"].apply(clean_isbn)

    books["Book-Title"] = books["Book-Title"].fillna("Unknown")
    books["Book-Author"] = books["Book-Author"].fillna("Unknown")

    # Menggunakan rating eksplisit saja (rating > 0)
    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"], errors="coerce"
    )
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    ratings_explicit = ratings[ratings["Book-Rating"] > 0].copy()

    # Memilih pengguna dengan minimal 5 rating dan buku dengan minimal 20 rating
    user_counts = ratings_explicit["User-ID"].value_counts()
    book_counts = ratings_explicit["ISBN"].value_counts()

    active_users = user_counts[user_counts >= 5].index
    popular_books = book_counts[book_counts >= 20].index

    ratings_filtered = ratings_explicit[
        ratings_explicit["User-ID"].isin(active_users)
        & ratings_explicit["ISBN"].isin(popular_books)
    ].copy()

    if ratings_filtered.empty:
        raise ValueError(
            "Data hasil filtering kosong. Periksa dataset atau kriteria minimal rating."
        )

    # Menghindari duplikasi pasangan pengguna dan buku
    ratings_filtered = (
        ratings_filtered.groupby(["User-ID", "ISBN"], as_index=False)["Book-Rating"]
        .mean()
    )

    # Membuat indeks pengguna dan buku
    user_codes, user_ids = pd.factorize(ratings_filtered["User-ID"])
    book_codes, book_ids = pd.factorize(ratings_filtered["ISBN"])

    # Membentuk matriks User-Item sparse
    user_item_matrix = csr_matrix(
        (
            ratings_filtered["Book-Rating"].to_numpy(),
            (user_codes, book_codes)
        ),
        shape=(len(user_ids), len(book_ids))
    )

    # Mengubah matriks User-Item menjadi Item-User
    item_user_matrix = user_item_matrix.T.tocsr()

    # Melatih KNN menggunakan Cosine Similarity
    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(11, item_user_matrix.shape[0])
    )
    model.fit(item_user_matrix)

    # Menggabungkan informasi buku berdasarkan ISBN
    books_info = (
        books[["ISBN", "Book-Title", "Book-Author"]]
        .drop_duplicates(subset="ISBN")
    )

    return (
        user_ids,
        book_ids,
        user_item_matrix,
        item_user_matrix,
        model,
        books_info,
        len(ratings_explicit),
        len(ratings_filtered)
    )


def recommend_books(
    user_id,
    user_ids,
    book_ids,
    user_item_matrix,
    item_user_matrix,
    model,
    books_info,
    n_recommendations=10
):
    """Menghasilkan rekomendasi buku untuk satu pengguna."""
    matching_users = np.where(user_ids == user_id)[0]

    if len(matching_users) == 0:
        return None, "User ID tidak ditemukan dalam data model."

    user_idx = matching_users[0]

    # Mengambil buku yang pernah diberi rating oleh pengguna
    user_ratings = user_item_matrix[user_idx].toarray().ravel()
    rated_books = np.where(user_ratings > 0)[0]

    if len(rated_books) == 0:
        return None, "Pengguna belum memiliki rating buku yang dapat digunakan."

    # Mencari buku yang mirip dengan buku yang pernah dinilai
    distances, indices = model.kneighbors(
        item_user_matrix[rated_books],
        n_neighbors=min(11, item_user_matrix.shape[0])
    )

    scores = {}
    similarity_totals = {}

    # Menghitung skor rekomendasi berbobot kemiripan
    for i, book_idx in enumerate(rated_books):
        rating = user_ratings[book_idx]

        for distance, similar_idx in zip(distances[i], indices[i]):
            similarity = 1 - distance

            # Mengabaikan buku yang sama, kemiripan nol, dan buku yang sudah dinilai
            if similar_idx == book_idx or similarity <= 0:
                continue
            if user_ratings[similar_idx] > 0:
                continue

            scores[similar_idx] = (
                scores.get(similar_idx, 0) + similarity * rating
            )
            similarity_totals[similar_idx] = (
                similarity_totals.get(similar_idx, 0) + similarity
            )

    if not scores:
        return None, "Belum ada buku yang dapat direkomendasikan untuk pengguna ini."

    recommendations = []
    for book_idx, score in scores.items():
        final_score = score / similarity_totals[book_idx]
        recommendations.append((book_ids[book_idx], final_score))

    recommendations = sorted(
        recommendations,
        key=lambda item: item[1],
        reverse=True
    )[:n_recommendations]

    result = pd.DataFrame(
        recommendations,
        columns=["ISBN", "Recommendation-Score"]
    )

    result = result.merge(books_info, on="ISBN", how="left")
    result["Book-Title"] = result["Book-Title"].fillna("Informasi judul tidak tersedia")
    result["Book-Author"] = result["Book-Author"].fillna("Informasi penulis tidak tersedia")

    return result[
        ["ISBN", "Book-Title", "Book-Author", "Recommendation-Score"]
    ], None


st.header("1. Unggah Dataset")
st.caption(
    "Unggah file CSV Book-Crossing. File harus menggunakan pemisah titik koma (;). "
    "Dataset Users bersifat opsional karena tidak digunakan langsung dalam pembentukan model."
)

col1, col2 = st.columns(2)
with col1:
    books_upload = st.file_uploader(
        "Dataset Books (BX-Books.csv)",
        type=["csv"],
        key="books"
    )
with col2:
    ratings_upload = st.file_uploader(
        "Dataset Ratings (BX-Book-Ratings.csv)",
        type=["csv"],
        key="ratings"
    )

users_upload = st.file_uploader(
    "Dataset Users (BX-Users.csv) — opsional",
    type=["csv"],
    key="users"
)

if books_upload is not None and ratings_upload is not None:
    if st.button("Bangun Model Rekomendasi", type="primary"):
        try:
            # Cache resource berdasarkan isi file agar model tidak dibangun ulang setiap interaksi.
            model_data = build_recommendation_model(
                books_upload.getvalue(),
                ratings_upload.getvalue()
            )
            st.session_state["model_data"] = model_data
            st.success("Model Collaborative Filtering berhasil dibuat!")
        except Exception as error:
            st.error(f"Gagal membangun model: {error}")
else:
    st.info("Silakan unggah dataset Books dan Ratings terlebih dahulu.")

if "model_data" in st.session_state:
    (
        user_ids,
        book_ids,
        user_item_matrix,
        item_user_matrix,
        model,
        books_info,
        explicit_count,
        filtered_count
    ) = st.session_state["model_data"]

    st.header("2. Informasi Model")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Pengguna dalam model", f"{len(user_ids):,}")
    metric2.metric("Buku dalam model", f"{len(book_ids):,}")
    metric3.metric("Rating setelah filtering", f"{filtered_count:,}")

    st.caption(f"Jumlah rating eksplisit sebelum filtering: {explicit_count:,}")

    st.header("3. Cari Rekomendasi Buku")
    available_user_ids = sorted(user_ids.tolist())

    selected_user = st.selectbox(
        "Pilih User ID",
        options=available_user_ids
    )
    number_of_recommendations = st.slider(
        "Jumlah rekomendasi buku",
        min_value=1,
        max_value=10,
        value=10
    )

    if st.button("Tampilkan Rekomendasi", type="primary"):
        result, error_message = recommend_books(
            selected_user,
            user_ids,
            book_ids,
            user_item_matrix,
            item_user_matrix,
            model,
            books_info,
            number_of_recommendations
        )

        if error_message:
            st.warning(error_message)
        else:
            st.success(f"Rekomendasi buku untuk User ID {selected_user}")
            result = result.copy()
            result["Recommendation-Score"] = result["Recommendation-Score"].round(4)
            st.dataframe(result, use_container_width=True, hide_index=True)

            csv = result.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Unduh Hasil Rekomendasi (CSV)",
                data=csv,
                file_name=f"rekomendasi_user_{selected_user}.csv",
                mime="text/csv"
            )

st.divider()
st.caption("Book Recommendation System | Item-Based Collaborative Filtering")
