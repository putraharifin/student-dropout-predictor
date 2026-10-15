# -*- coding: utf-8 -*-
"""
Prototype prediksi dropout mahasiswa - Jaya Jaya Institut.

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Dropout | Jaya Jaya Institut",
                   page_icon="🎓", layout="wide")

MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"
# Model biner: kode 1 = Dropout, kode 0 = Graduate. Mahasiswa Enrolled tidak dipakai untuk melatih model.
# Aplikasi ini menilai potensi Dropout atau Graduate, termasuk untuk mahasiswa yang masih aktif (Enrolled).
LABEL_HASIL = {
    "Dropout": "Berpotensi DROPOUT (berisiko putus kuliah)",
    "Graduate": "Berpotensi GRADUATE (diprediksi lulus)",
}


# --------------------------------------------------------------------------
# Muat model dan fungsi bantu
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def muat_bundle():
    return joblib.load(MODEL_PATH)


def tambah_fitur(data: pd.DataFrame) -> pd.DataFrame:
    """Fitur turunan. Harus sama persis dengan fungsi di notebook.ipynb."""
    out = data.copy()
    for s, nama in [("1st", "sem1"), ("2nd", "sem2")]:
        enrolled = out[f"Curricular_units_{s}_sem_enrolled"]
        approved = out[f"Curricular_units_{s}_sem_approved"]
        out[f"approval_rate_{nama}"] = np.where(enrolled > 0, approved / enrolled.where(enrolled > 0, 1), 0.0)
    out["total_approved"] = (out["Curricular_units_1st_sem_approved"]
                             + out["Curricular_units_2nd_sem_approved"])
    out["total_enrolled"] = (out["Curricular_units_1st_sem_enrolled"]
                             + out["Curricular_units_2nd_sem_enrolled"])
    out["grade_trend"] = (out["Curricular_units_2nd_sem_grade"]
                          - out["Curricular_units_1st_sem_grade"])
    out["approved_trend"] = (out["Curricular_units_2nd_sem_approved"]
                             - out["Curricular_units_1st_sem_approved"])
    return out


def prediksi(model, data_mentah: pd.DataFrame) -> np.ndarray:
    """Mengembalikan peluang Dropout (kelas 1) untuk setiap baris."""
    return model.predict_proba(tambah_fitur(data_mentah))[:, 1]


def tingkat_risiko(p_dropout: float) -> str:
    if p_dropout >= 0.60:
        return "Tinggi"
    if p_dropout >= 0.30:
        return "Sedang"
    return "Rendah"


try:
    bundle = muat_bundle()
except FileNotFoundError:
    st.error("File model tidak ditemukan. Jalankan notebook.ipynb sampai selesai "
             "agar file `model/model.joblib` terbentuk.")
    st.stop()

model = bundle["model"]
AMBANG = float(bundle.get("ambang", 0.5))
FITUR = bundle["fitur_input"]
DEFAULT = bundle["defaults"]
OPSI = bundle["opsi_nominal"]
RENTANG = bundle["rentang"]
KAMUS = bundle["labels"]

KOLOM_INT = [
    "Application_order", "Age_at_enrollment",
    *[f"Curricular_units_{s}_sem_{k}" for s in ("1st", "2nd")
      for k in ("credited", "enrolled", "evaluations", "approved", "without_evaluations")],
]
KOLOM_FLOAT = ["Previous_qualification_grade", "Admission_grade",
               "Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade",
               "Unemployment_rate", "Inflation_rate", "GDP"]

# --------------------------------------------------------------------------
# Contoh profil mahasiswa
# --------------------------------------------------------------------------
CONTOH = {
    "Risiko tinggi: menunggak, nilai rendah": {
        "Tuition_fees_up_to_date": 0, "Debtor": 1, "Scholarship_holder": 0,
        "Age_at_enrollment": 27, "Gender": 1,
        "Curricular_units_1st_sem_enrolled": 6, "Curricular_units_1st_sem_evaluations": 6,
        "Curricular_units_1st_sem_approved": 1, "Curricular_units_1st_sem_grade": 8.0,
        "Curricular_units_2nd_sem_enrolled": 6, "Curricular_units_2nd_sem_evaluations": 4,
        "Curricular_units_2nd_sem_approved": 0, "Curricular_units_2nd_sem_grade": 0.0,
    },
    "Risiko rendah: lancar, penerima beasiswa": {
        "Tuition_fees_up_to_date": 1, "Debtor": 0, "Scholarship_holder": 1,
        "Age_at_enrollment": 19, "Gender": 0,
        "Curricular_units_1st_sem_enrolled": 6, "Curricular_units_1st_sem_evaluations": 7,
        "Curricular_units_1st_sem_approved": 6, "Curricular_units_1st_sem_grade": 14.0,
        "Curricular_units_2nd_sem_enrolled": 6, "Curricular_units_2nd_sem_evaluations": 7,
        "Curricular_units_2nd_sem_approved": 6, "Curricular_units_2nd_sem_grade": 14.0,
    },
    "Sedang: bayar lancar, 4 dari 6 MK lulus": {
        "Tuition_fees_up_to_date": 1, "Debtor": 0, "Scholarship_holder": 0,
        "Age_at_enrollment": 21, "Gender": 1,
        "Curricular_units_1st_sem_enrolled": 6, "Curricular_units_1st_sem_evaluations": 8,
        "Curricular_units_1st_sem_approved": 4, "Curricular_units_1st_sem_grade": 13.0,
        "Curricular_units_2nd_sem_enrolled": 6, "Curricular_units_2nd_sem_evaluations": 8,
        "Curricular_units_2nd_sem_approved": 4, "Curricular_units_2nd_sem_grade": 13.0,
    },
}


def nilai_awal(kol):
    v = DEFAULT[kol]
    if kol in KOLOM_INT:
        return int(round(v))
    if kol in KOLOM_FLOAT:
        return float(v)
    return int(v)


for kol in FITUR:
    st.session_state.setdefault(kol, nilai_awal(kol))
st.session_state.setdefault("contoh", "(isi manual)")


def terapkan_contoh():
    pilihan = st.session_state["contoh"]
    if pilihan in CONTOH:
        for k, v in CONTOH[pilihan].items():
            st.session_state[k] = v


# --------------------------------------------------------------------------
# Komponen input (sidebar)
# --------------------------------------------------------------------------
def input_biner(label, kol, teks0, teks1, help=None):
    return st.radio(label, [0, 1], key=kol, horizontal=True, help=help,
                    format_func=lambda v: teks1 if v == 1 else teks0)


def input_kode(label, kol, kamus=None, help=None):
    kamus = kamus or {}
    return st.selectbox(label, OPSI[kol], key=kol, help=help,
                        format_func=lambda v: f"{kamus[v]}" if v in kamus else f"Kode {v}")


def input_int(label, kol, minimum=0, maksimum=None, help=None):
    mx = int(maksimum if maksimum is not None else max(RENTANG[kol][1], 1))
    return st.number_input(label, min_value=minimum, max_value=mx, step=1, key=kol, help=help)


def input_float(label, kol, minimum, maksimum, langkah=0.1, help=None):
    return st.number_input(label, min_value=float(minimum), max_value=float(maksimum),
                           step=float(langkah), key=kol, help=help)


st.sidebar.title("🎓 Input data mahasiswa")
st.sidebar.selectbox("Muat contoh profil", ["(isi manual)"] + list(CONTOH),
                     key="contoh", on_change=terapkan_contoh,
                     help="Pilih contoh untuk mengisi form dengan cepat. Nilai tetap bisa diubah.")

with st.sidebar.expander("👤 Profil dan pendaftaran", expanded=True):
    input_int("Usia saat mendaftar", "Age_at_enrollment", 15, 80)
    input_biner("Jenis kelamin", "Gender", "Perempuan", "Laki-laki")
    input_kode("Status pernikahan", "Marital_status", KAMUS["marital"])
    input_kode("Program studi", "Course", KAMUS["course"])
    input_biner("Waktu kuliah", "Daytime_evening_attendance", "Malam", "Siang")
    input_biner("Mahasiswa perantau?", "Displaced", "Tidak", "Ya")
    input_kode("Jalur pendaftaran", "Application_mode", KAMUS["app_mode"])
    input_int("Urutan pilihan jurusan (0 = pilihan pertama)", "Application_order", 0, 9)
    input_kode("Pendidikan sebelumnya", "Previous_qualification", KAMUS["prev_qual"])
    input_float("Nilai pendidikan sebelumnya (0-200)", "Previous_qualification_grade", 0, 200)
    input_float("Nilai masuk / admission grade (0-200)", "Admission_grade", 0, 200)

with st.sidebar.expander("💰 Keuangan", expanded=True):
    input_biner("Uang kuliah", "Tuition_fees_up_to_date", "Menunggak", "Tepat waktu")
    input_biner("Punya tunggakan utang (debtor)?", "Debtor", "Tidak", "Ya")
    input_biner("Penerima beasiswa?", "Scholarship_holder", "Tidak", "Ya")

with st.sidebar.expander("📘 Akademik semester 1", expanded=True):
    input_int("MK diakui (credited)", "Curricular_units_1st_sem_credited")
    input_int("MK diambil (enrolled)", "Curricular_units_1st_sem_enrolled")
    input_int("Jumlah evaluasi", "Curricular_units_1st_sem_evaluations")
    input_int("MK lulus (approved)", "Curricular_units_1st_sem_approved")
    input_float("Nilai rata-rata (0-20)", "Curricular_units_1st_sem_grade", 0, 20)
    input_int("MK tanpa evaluasi", "Curricular_units_1st_sem_without_evaluations")

with st.sidebar.expander("📗 Akademik semester 2", expanded=True):
    input_int("MK diakui (credited)", "Curricular_units_2nd_sem_credited")
    input_int("MK diambil (enrolled)", "Curricular_units_2nd_sem_enrolled")
    input_int("Jumlah evaluasi", "Curricular_units_2nd_sem_evaluations")
    input_int("MK lulus (approved)", "Curricular_units_2nd_sem_approved")
    input_float("Nilai rata-rata (0-20)", "Curricular_units_2nd_sem_grade", 0, 20)
    input_int("MK tanpa evaluasi", "Curricular_units_2nd_sem_without_evaluations")

with st.sidebar.expander("⚙️ Data tambahan (opsional)"):
    st.caption("Nilai awal berasal dari nilai paling umum di data latih. "
               "Pengaruh fitur ini kecil.")
    input_kode("Kewarganegaraan (kode)", "Nacionality")
    input_biner("Mahasiswa internasional?", "International", "Tidak", "Ya")
    input_biner("Kebutuhan pendidikan khusus?", "Educational_special_needs", "Tidak", "Ya")
    input_kode("Pendidikan ibu (kode)", "Mothers_qualification")
    input_kode("Pendidikan ayah (kode)", "Fathers_qualification")
    input_kode("Pekerjaan ibu (kode)", "Mothers_occupation")
    input_kode("Pekerjaan ayah (kode)", "Fathers_occupation")
    input_float("Tingkat pengangguran (%)", "Unemployment_rate", RENTANG["Unemployment_rate"][0] - 5,
                RENTANG["Unemployment_rate"][1] + 5)
    input_float("Tingkat inflasi (%)", "Inflation_rate", RENTANG["Inflation_rate"][0] - 5,
                RENTANG["Inflation_rate"][1] + 5)
    input_float("GDP (pertumbuhan, %)", "GDP", RENTANG["GDP"][0] - 5, RENTANG["GDP"][1] + 5, 0.01)

# --------------------------------------------------------------------------
# Halaman utama
# --------------------------------------------------------------------------
st.title("🎓 Prediksi Dropout Mahasiswa")
st.caption("Jaya Jaya Institut · Alat bantu keputusan untuk deteksi dini mahasiswa berisiko. "
           "Hasil prediksi bukan keputusan final.")

tab_satu, tab_massal, tab_info = st.tabs(["🔍 Prediksi satu mahasiswa", "📂 Prediksi massal (CSV)", "ℹ️ Tentang model"])

# ---- Tab 1: satu mahasiswa ------------------------------------------------
with tab_satu:
    baris = {k: st.session_state[k] for k in FITUR}
    df_input = pd.DataFrame([baris])[FITUR]
    p_dropout = float(prediksi(model, df_input)[0])
    p_graduate = 1 - p_dropout
    kelas = "Dropout" if p_dropout >= AMBANG else "Graduate"
    risiko = tingkat_risiko(p_dropout)

    for sem, nama in [("1st", "semester 1"), ("2nd", "semester 2")]:
        if baris[f"Curricular_units_{sem}_sem_approved"] > baris[f"Curricular_units_{sem}_sem_enrolled"]:
            st.warning(f"MK lulus lebih banyak daripada MK diambil di {nama}. Periksa kembali input Anda.")

    st.subheader("Hasil prediksi")
    if kelas == "Dropout":
        st.error(f"### 🔴 {LABEL_HASIL[kelas]}")
    else:
        st.success(f"### 🟢 {LABEL_HASIL[kelas]}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Peluang Dropout", f"{p_dropout:.1%}")
    c2.metric("Peluang Graduate", f"{p_graduate:.1%}")
    c3.metric("Tingkat risiko dropout", risiko)

    st.bar_chart(pd.DataFrame({"Peluang": [p_dropout, p_graduate]}, index=["Dropout", "Graduate"]),
                 horizontal=True, height=160)

    if max(p_dropout, p_graduate) < 0.70:
        st.info("Keyakinan model di bawah 70%. Pada data uji, prediksi seperti ini jauh kurang akurat. "
                "Tinjau mahasiswa ini secara manual.")
    st.caption("Untuk mahasiswa berstatus Enrolled, hasil ini adalah **potensi** di masa mendatang, bukan kepastian.")

    # Sinyal peringatan berbasis temuan EDA
    turunan = tambah_fitur(df_input).iloc[0]
    sinyal = []
    if baris["Tuition_fees_up_to_date"] == 0:
        sinyal.append(("Uang kuliah menunggak", "Di data, sekitar 87% mahasiswa penunggak dropout.",
                       "Tawarkan cicilan atau restrukturisasi biaya."))
    if baris["Debtor"] == 1:
        sinyal.append(("Punya tunggakan utang", "Tingkat dropout debitur sekitar 62%.",
                       "Hubungi bagian keuangan untuk skema bantuan."))
    if baris["Scholarship_holder"] == 0 and (baris["Tuition_fees_up_to_date"] == 0 or baris["Debtor"] == 1):
        sinyal.append(("Bermasalah finansial tanpa beasiswa", "Penerima beasiswa dropout sekitar 12%, non-penerima 39%.",
                       "Nilai kelayakan beasiswa atau bantuan biaya."))
    if turunan["approval_rate_sem2"] < 0.5:
        sinyal.append(("Kurang dari separuh MK lulus di semester 2", "Rasio kelulusan semester 2 adalah fitur terpenting model.",
                       "Jadwalkan bimbingan akademik dan kelas remedial."))
    if baris["Curricular_units_2nd_sem_approved"] == 0:
        sinyal.append(("Tidak ada MK lulus di semester 2", "Sekitar 51% mahasiswa dropout berada di kondisi ini.",
                       "Hubungi mahasiswa segera untuk konseling."))
    if turunan["grade_trend"] <= -2:
        sinyal.append(("Nilai turun tajam dari semester 1 ke 2", f"Perubahan nilai: {turunan['grade_trend']:.1f} poin.",
                       "Gali penyebab (beban kerja, kesehatan, motivasi)."))
    if baris["Age_at_enrollment"] >= 23:
        sinyal.append(("Mahasiswa dewasa (23 tahun ke atas)", "Dropout naik di atas 43% pada kelompok usia ini.",
                       "Tawarkan jadwal fleksibel atau kelas malam."))

    st.subheader("Sinyal peringatan dan tindak lanjut")
    if sinyal:
        st.dataframe(pd.DataFrame(sinyal, columns=["Sinyal", "Alasan (dari data)", "Saran tindak lanjut"]),
                     hide_index=True, width="stretch")
    else:
        st.write("Tidak ada sinyal peringatan utama pada data yang dimasukkan.")

    with st.expander("Lihat fitur turunan yang dihitung otomatis"):
        st.dataframe(turunan[["approval_rate_sem1", "approval_rate_sem2", "total_approved",
                              "total_enrolled", "grade_trend", "approved_trend"]].to_frame("Nilai").round(3))

# ---- Tab 2: CSV -----------------------------------------------------------
with tab_massal:
    st.write("Unggah file CSV berisi data mahasiswa. Kolom harus sama dengan dataset asli "
             "(pemisah `;` atau `,`). Kolom `Status` boleh ada atau tidak.")
    berkas = st.file_uploader("Pilih file CSV", type=["csv"])
    if berkas is not None:
        try:
            data = pd.read_csv(berkas, sep=None, engine="python", encoding="utf-8-sig")
            data.columns = data.columns.str.strip()
            hilang = [c for c in FITUR if c not in data.columns]
            if hilang:
                st.error(f"Kolom berikut tidak ada di file: {', '.join(hilang)}")
            else:
                p_drop = prediksi(model, data[FITUR])
                hasil = data.copy()
                hasil["Peluang_Dropout"] = p_drop.round(4)
                hasil["Peluang_Graduate"] = (1 - p_drop).round(4)
                hasil["Prediksi"] = np.where(p_drop >= AMBANG, "Dropout", "Graduate")
                hasil["Tingkat_Risiko"] = [tingkat_risiko(v) for v in p_drop]

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Jumlah mahasiswa", f"{len(hasil):,}")
                k2.metric("Risiko tinggi", f"{(hasil['Tingkat_Risiko'] == 'Tinggi').sum():,}")
                k3.metric("Risiko sedang", f"{(hasil['Tingkat_Risiko'] == 'Sedang').sum():,}")
                k4.metric("Risiko rendah", f"{(hasil['Tingkat_Risiko'] == 'Rendah').sum():,}")

                st.bar_chart(hasil["Prediksi"].value_counts().reindex(["Dropout", "Graduate"], fill_value=0))
                if "Status" in hasil.columns:
                    st.write("Sebaran tingkat risiko menurut kolom `Status` di file:")
                    st.dataframe(pd.crosstab(hasil["Tingkat_Risiko"], hasil["Status"])
                                 .reindex(["Tinggi", "Sedang", "Rendah"]).fillna(0).astype(int))
                tampil = ["Peluang_Dropout", "Prediksi", "Tingkat_Risiko"]
                st.dataframe(hasil.sort_values("Peluang_Dropout", ascending=False)[tampil + FITUR[:8]],
                             width="stretch", height=380)
                st.download_button("⬇️ Unduh hasil prediksi (CSV)",
                                   hasil.to_csv(index=False).encode("utf-8"),
                                   file_name="hasil_prediksi_dropout.csv", mime="text/csv")
                st.caption("Mahasiswa berstatus Enrolled boleh ikut diunggah. Hasilnya adalah potensi, bukan kepastian. "
                           "Data berlabel yang sama dengan data latih akan terlihat sangat akurat. "
                           "Gunakan data kohort baru untuk evaluasi yang jujur.")
        except Exception as e:  # noqa: BLE001
            st.error(f"File tidak dapat diproses: {e}")

# ---- Tab 3: info ----------------------------------------------------------
with tab_info:
    m = bundle["metrics_test"]
    st.subheader(f"Model: {bundle['model_name']} (klasifikasi biner)")
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy (data uji)", f"{m['Accuracy']:.1%}")
    b.metric("Precision Dropout", f"{m['Precision (Dropout)']:.1%}")
    c.metric("Recall Dropout", f"{m['Recall (Dropout)']:.1%}")
    d.metric("F1 Dropout", f"{m['F1 (Dropout)']:.3f}")
    e.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    st.markdown(
        """
**Cara kerja.** Model membaca data pendaftaran, keuangan, dan kinerja akademik semester 1-2.
Model lalu menghitung peluang mahasiswa **Dropout** atau **Graduate**.

**Data pelatihan.** Model hanya dilatih dengan mahasiswa berstatus `Dropout` dan `Graduate`
(3.630 mahasiswa). Mahasiswa `Enrolled` belum punya hasil akhir, sehingga tidak dipakai untuk melatih model.
Mereka dipakai sebagai data prediksi: model menilai **potensi** mereka untuk dropout atau lulus.

**Cara membaca hasil.**
- Peluang dropout 50% atau lebih menghasilkan prediksi **Dropout**. Di bawah itu, prediksi **Graduate**.
- Tingkat risiko: 60% atau lebih = **tinggi**, 30-60% = **sedang**, di bawah 30% = **rendah**.
- Pada data berlabel, kelompok risiko tinggi benar-benar dropout sekitar 93%, sedang 38%, dan rendah 8%.

**Keterbatasan.**
- Model memakai data satu institusi. Uji ulang sebelum dipakai pada kohort baru.
- Model menunjukkan hubungan, bukan sebab-akibat.
- Prediksi untuk mahasiswa `Enrolled` tidak dapat divalidasi, karena label akhirnya belum ada.
- Aplikasi ini memakai model penuh (data semester 1 dan 2). Jangan mengisi semester 2 dengan angka 0
  bila datanya belum ada. Model membaca angka 0 sebagai tanda dropout yang kuat.
        """
    )
