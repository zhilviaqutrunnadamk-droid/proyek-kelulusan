
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Pengaturan halaman
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓"
)

# Membuat dataset simulasi
np.random.seed(42)

data = pd.DataFrame({
    "Kehadiran": np.random.randint(40, 101, 500),
    "Tugas": np.random.randint(40, 101, 500),
    "UTS": np.random.randint(40, 101, 500),
    "UAS": np.random.randint(40, 101, 500)
})

# Menentukan label kelulusan untuk data simulasi
nilai = (
    data["Kehadiran"] * 0.10 +
    data["Tugas"] * 0.20 +
    data["UTS"] * 0.30 +
    data["UAS"] * 0.40
)

data["Status"] = np.where(nilai >= 60, 1, 0)

# Melatih algoritma Random Forest
X = data[["Kehadiran", "Tugas", "UTS", "UAS"]]
y = data["Status"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Tampilan website
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk mengetahui hasil prediksi kelulusan.")

st.divider()

# Form input pengguna
nama = st.text_input("Nama Pengguna")

kehadiran = st.number_input(
    "Nilai Kehadiran", 0, 100, 80
)

tugas = st.number_input(
    "Nilai Tugas", 0, 100, 80
)

uts = st.number_input(
    "Nilai UTS", 0, 100, 80
)

uas = st.number_input(
    "Nilai UAS", 0, 100, 80
)

# Tombol prediksi
if st.button("Prediksi Kelulusan"):

    if nama.strip() == "":
        st.warning("Silakan isi nama pengguna terlebih dahulu.")

    else:
        input_data = pd.DataFrame(
            [[kehadiran, tugas, uts, uas]],
            columns=["Kehadiran", "Tugas", "UTS", "UAS"]
        )

        hasil = model.predict(input_data)[0]
        probabilitas = model.predict_proba(input_data)[0][hasil] * 100

        st.divider()
        st.subheader("Hasil Prediksi")

        st.write("Nama Mahasiswa:", nama)

        if hasil == 1:
            st.success("SELAMAT! DIPREDIKSI LULUS 🎉")
        else:
            st.error("DIPREDIKSI BELUM LULUS")

        st.write("Tingkat keyakinan model:", round(probabilitas, 2), "%")

        st.caption(
            "Hasil merupakan simulasi model AI, bukan keputusan kelulusan resmi."
        )
