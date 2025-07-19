import streamlit as st
import pandas as pd
import joblib # Menggunakan joblib untuk memuat model

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Keberhasilan Peserta Literasi Digital",
    page_icon="💡", # Menggunakan ikon bohlam untuk literasi
    layout="centered",
    initial_sidebar_state="auto"
)

# --- CSS Kustom untuk Background dan Teks ---
st.markdown(
    """
    <style>
    /* Mengubah warna background */
    .stApp {
        background-color: #F0F2F6; /* Warna abu-abu terang yang aesthetic dan lembut */
    }

    /* Mengatur ukuran font untuk subheader agar lebih ringkas */
    h4 {
        font-size: 1.2em; /* Mengurangi ukuran font agar lebih mungkin satu baris */
        margin-bottom: 0.5rem; /* Sedikit spasi di bawahnya */
    }

    /* Styling untuk tombol prediksi */
    div.stButton > button {
        background-color: #4CAF50; /* Warna hijau */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #45a049; /* Warna hijau sedikit lebih gelap saat hover */
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Judul dan Deskripsi Aplikasi ---
st.title("📚 Prediksi Keberhasilan Peserta Pelatihan Literasi Digital")
st.markdown("""
Aplikasi ini memprediksi apakah seorang peserta pelatihan akan **berhasil** atau **gagal**
dalam meningkatkan literasi digitalnya berdasarkan skor post-training dan pre-training
untuk computer knowledge, mobile literacy, dan internet usage.
""")

st.write("---")

# --- Memuat Model ---
# Menggunakan @st.cache_resource agar model hanya dimuat sekali
@st.cache_resource
def load_model():
    try:
        # Menggunakan joblib.load() untuk memuat model
        model = joblib.load('digital_literacy_baru.pkl')
        st.success("✅ Model 'digital_literacy_baru.pkl' berhasil dimuat!")
        return model
    except FileNotFoundError:
        st.error("❌ Model 'digital_literacy_baru.pkl' tidak ditemukan. Pastikan file model berada di direktori yang sama dengan 'app.py'.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Panggil fungsi untuk memuat model
knn_model = load_model()

# Lanjutkan hanya jika model berhasil dimuat
if knn_model is not None:
    st.subheader("📝 Masukkan Data Peserta")

    # --- Input Pengguna dengan Slider Numerik ---
    # Menggunakan kolom untuk tata letak yang lebih rapi
    col1, col2 = st.columns(2)

    with col1: # Kolom KIRI untuk skor Pre-training
        st.markdown("#### Skor Sebelum Pelatihan (Pre-training)")
        new_BCKS = st.slider(
            "Basic Computer Knowledge Score",
            min_value=0.0, max_value=50.0, value=25.0, step=0.1, # Max 50, nilai awal 25
            help="Skor pengetahuan dasar komputer sebelum pelatihan (skala 0-50)."
        )
        new_IUS = st.slider(
            "Internet Usage Score",
            min_value=0.0, max_value=50.0, value=25.0, step=0.1, # Max 50, nilai awal 25
            help="Skor penggunaan internet sebelum pelatihan (skala 0-50)."
        )
        new_MLS = st.slider(
            "Mobile Literacy Score",
            min_value=0.0, max_value=50.0, value=25.0, step=0.1, # Max 50, nilai awal 25
            help="Skor literasi perangkat mobile sebelum pelatihan (skala 0-50)."
        )
    with col2: # Kolom KANAN untuk skor Post Training
        st.markdown("#### Skor Setelah Pelatihan (Post-training)")
        new_PTBCKS = st.slider(
            "Post Training Basic Computer Knowledge Score",
            min_value=0.0, max_value=100.0, value=50.0, step=0.1, # Max 100, nilai awal 50
            help="Skor pengetahuan dasar komputer setelah pelatihan (skala 0-100)."
        )
        new_PTIUS = st.slider(
            "Post Training Internet Usage Score",
            min_value=0.0, max_value=100.0, value=50.0, step=0.1, # Max 100, nilai awal 50
            help="Skor penggunaan internet setelah pelatihan (skala 0-100)."
        )
        new_PTMLS = st.slider(
            "Post Training Mobile Literacy Score",
            min_value=0.0, max_value=100.0, value=50.0, step=0.1, # Max 100, nilai awal 50
            help="Skor literasi perangkat mobile setelah pelatihan (skala 0-100)."
        )

    # --- Tombol Prediksi ---
    st.write("---")
    if st.button("🚀 Prediksi Keberhasilan"):
        # Buat DataFrame dari input baru
        # PENTING: Urutan nilai di sini harus sesuai dengan urutan kolom yang diharapkan oleh model Anda.
        # Jika model Anda dilatih dengan urutan 'Post_Training_Basic_Computer_Knowledge_Score',
        # 'Post_Training_Mobile_Literacy_Score', dll., maka pastikan urutan ini benar.
        new_data_df = pd.DataFrame(
            [[new_PTBCKS, new_PTMLS, new_PTIUS, new_BCKS, new_MLS, new_IUS]],
            columns=[
                'Post_Training_Basic_Computer_Knowledge_Score',
                'Post_Training_Mobile_Literacy_Score',
                'Post_Training_Internet_Usage_Score',
                'Basic_Computer_Knowledge_Score',
                'Mobile_Literacy_Score',
                'Internet_Usage_Score'
            ]
        )

        try:
            # Lakukan prediksi pada data input mentah (tanpa scaling)
            predicted_code = knn_model.predict(new_data_df)[0]

            # Konversi hasil prediksi ke label asli
            label_mapping = {1: 'Berhasil 🎉', 0: 'Gagal 🙁'}
            predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

            st.markdown(f"### Hasil Prediksi: **{predicted_label}**")
            if predicted_code == 1:
                st.balloons() # Efek visual menarik jika berhasil
                st.write("Selamat! Peserta ini diprediksi berhasil meningkatkan literasi digitalnya. 👍")
            else:
                st.write("Sayangnya, peserta ini diprediksi gagal. Mungkin diperlukan intervensi atau dukungan lebih lanjut. 😔")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}. Pastikan format input dan model sesuai.")

    st.write("---")
    st.info("Aplikasi ini dibuat oleh Fitry Rosita Desviani.")
