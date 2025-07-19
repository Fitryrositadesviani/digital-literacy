import streamlit as st
import pandas as pd
import joblib

# --- Judul dan Deskripsi Aplikasi ---
st.set_page_config(page_title="Prediksi Keberhasilan Literasi Digital", layout="centered")

st.title("💡 Prediksi Keberhasilan Peserta Pelatihan Literasi Digital")
st.markdown("""
Aplikasi ini membantu memprediksi **keberhasilan** atau **kegagalan** seorang peserta dalam pelatihan literasi digital berdasarkan beberapa skor penting.
""")

st.write("---")

# --- Memuat Model ---
# Pastikan file model 'digital_literacy_baru.pkl' berada di direktori yang sama dengan app.py
try:
    model = joblib.load('digital_literacy_baru.pkl')
    st.success("Model prediksi berhasil dimuat! Siap untuk memprediksi.")
except FileNotFoundError:
    st.error("Error: File model 'digital_literacy_baru.pkl' tidak ditemukan. Pastikan file model ada di direktori yang sama.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

st.subheader("📊 Masukkan Data Peserta Pelatihan")

# --- Input Pengguna ---
with st.form("prediction_form"):
    st.write("Silakan masukkan skor-skor berikut untuk memprediksi hasil:")

    col1, col2 = st.columns(2)
    
    with col1:
        new_PTBCKS = st.number_input("Skor Pengetahuan Komputer Dasar Pasca-Pelatihan (Post Training Basic Computer Knowledge Score)", min_value=0.0, max_value=100.0, value=50.0)
        new_PTMLS = st.number_input("Skor Literasi Seluler Pasca-Pelatihan (Post Training Mobile Literacy Score)", min_value=0.0, max_value=100.0, value=50.0)
        new_PTIUS = st.number_input("Skor Penggunaan Internet Pasca-Pelatihan (Post Training Internet Usage Score)", min_value=0.0, max_value=100.0, value=50.0)
    
    with col2:
        new_BCKS = st.number_input("Skor Pengetahuan Komputer Dasar (Basic Computer Knowledge Score)", min_value=0.0, max_value=100.0, value=50.0)
        new_MLS = st.number_input("Skor Literasi Seluler (Mobile Literacy Score)", min_value=0.0, max_value=100.0, value=50.0)
        new_IUS = st.number_input("Skor Penggunaan Internet (Internet Usage Score)", min_value=0.0, max_value=100.0, value=50.0)

    submitted = st.form_submit_button("Prediksi Hasil")

# --- Logika Prediksi ---
if submitted:
    # Buat DataFrame dari input baru
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
        # Lakukan prediksi
        predicted_code = model.predict(new_data_df)[0]

        # Konversi hasil prediksi ke label asli
        label_mapping = {1: 'Berhasil', 0: 'Gagal'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.write("---")
        st.subheader("🌟 Hasil Prediksi")
        if predicted_label == 'Berhasil':
            st.success(f"Berdasarkan data yang Anda masukkan, peserta ini **diprediksi BERHASIL** dalam pelatihan literasi digital! 🎉")
        elif predicted_label == 'Gagal':
            st.warning(f"Berdasarkan data yang Anda masukkan, peserta ini **diprediksi GAGAL** dalam pelatihan literasi digital. 😔")
        else:
            st.info(f"Hasil prediksi: {predicted_label}")
        
        st.markdown("""
        <small>Prediksi ini adalah hasil dari model machine learning dan dapat digunakan sebagai panduan awal.</small>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.write("---")
st.info("Aplikasi ini dibuat untuk tujuan demonstrasi dan pembelajaran.")