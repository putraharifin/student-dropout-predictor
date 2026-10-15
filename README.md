# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institut

Proyek ini membangun sistem deteksi dini mahasiswa yang berisiko _dropout_. Sistem terdiri dari analisis data, model machine learning, prototype Streamlit, dan rancangan dashboard monitoring.

- **Nama:** Muhammad Putra Harifin Pane
- **Email:** putraharifin@gmail.com
- **Id Dicoding:** putraharifin25

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang berdiri sejak tahun 2000. Institusi ini telah meluluskan banyak alumni dengan reputasi baik. Namun banyak mahasiswa tidak menyelesaikan studinya atau _dropout_.

Dropout yang tinggi merugikan banyak pihak:

- Mahasiswa kehilangan waktu dan biaya.
- Institusi kehilangan pendapatan dan reputasi.
- Kapasitas kelas terpakai, tetapi tidak menghasilkan lulusan.

Pada dataset ini, **32,1%** mahasiswa (1.421 dari 4.424) berstatus _dropout_. Hampir 1 dari 3 mahasiswa tidak lulus. Institusi ingin mengenali mahasiswa berisiko **sedini mungkin**, lalu memberi bimbingan khusus sebelum terlambat.

## Permasalahan Bisnis

Tingginya angka dropout di Jaya Jaya Institut yang mencapai 32,1% (hampir 1 dari 3 mahasiswa) merupakan sebuah alarm bahaya yang mengancam stabilitas finansial dan reputasi institusi. Selama ini, pihak manajemen dan tim akademik hanya bertindak secara reaktif—mengetahui mahasiswa keluar setelah proses dropout resmi terjadi, sehingga bimbingan konseling atau intervensi akademik sudah terlambat untuk diberikan.

Akar masalah dari fenomena ini adalah ketiadaan visibilitas dan sistem deteksi dini yang mampu menyaring data historis mahasiswa secara real-time. Institusi belum mengetahui secara pasti bagaimana faktor finansial (seperti kelancaran pembayaran SPP dan status utang), faktor demografi (usia saat mendaftar, gender, latar belakang orang tua), serta performa akademik di awal semester saling berinteraksi memicu keputusan mahasiswa untuk berhenti kuliah. Akibatnya, alokasi sumber daya bimbingan konseling menjadi tidak tepat sasaran.

Oleh karena itu, permasalahan utama yang ingin diselesaikan melalui proyek data science ini adalah bagaimana mentransformasi data rekam jejak mahasiswa yang terpisah-pisah menjadi sebuah sistem prediksi otomatis dan dashboard monitoring. Sistem ini harus mampu mengidentifikasi mahasiswa yang berada di zona berisiko tinggi (high-risk) sedini mungkin—bahkan sejak akhir semester pertama studi mereka. Dengan begitu, tim akademik dapat melakukan intervensi bimbingan khusus yang terukur dan tepat waktu, yang pada akhirnya dapat menekan laju angka dropout dan menyelamatkan reputasi serta pendapatan institusi.

## Cakupan Proyek

**Yang dikerjakan:**

- Analisis data eksploratif (EDA) mendalam: distribusi, outlier, korelasi, uji statistik (Chi-square dan Kruskal-Wallis), serta analisis per faktor keuangan, akademik, demografi, dan program studi.
- Feature engineering dari kinerja akademik (rasio kelulusan, tren nilai).
- Model klasifikasi tiga kelas: `Dropout`, `Enrolled`, `Graduate`. Model yang dibandingkan: Logistic Regression, Random Forest, XGBoost, dan LightGBM.
- Analisis skenario deteksi dini (saat pendaftaran, akhir semester 1, akhir semester 2).
- Prototype aplikasi Streamlit untuk prediksi satu mahasiswa dan prediksi massal lewat CSV.
- Data siap pakai untuk dashboard (`dashboard_data.csv`) dan rancangan layout dashboard.
- Rekomendasi _action items_.

**Di luar cakupan:**

- Integrasi langsung dengan sistem akademik kampus.
- Uji coba intervensi di lapangan (A/B test).
- Analisis sebab-akibat. Proyek ini hanya menunjukkan hubungan antar variabel.

## Persiapan

**Sumber data:** [Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89) (UCI Machine Learning Repository). Pengumpul data: Realinho, Vieira Martins, Machado, dan Baptista (2021). Dataset berisi 4.424 mahasiswa dan 37 kolom. Tidak ada nilai kosong dan tidak ada baris duplikat. File lokal: `data.csv` (pemisah `;`).

**Struktur direktori:**

```
submission
├── model
│   ├── model.joblib          # model + metadata untuk Streamlit
│   └── metrics.json          # ringkasan metrik evaluasi
├── data.csv                  # dataset asli
├── dashboard_data.csv        # data berlabel + skor risiko untuk dashboard
├── notebook.ipynb            # EDA, preprocessing, modeling, evaluasi
├── app.py                    # prototype Streamlit
├── requirements.txt
├── README.md
└── putraharifin25-dashboard.png
```

**Setup environment:**  Python yang digunakan versi 3.11.

```bash
# 1. Buat virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Pasang dependensi
pip install -r requirements.txt
pip install jupyter        # hanya untuk menjalankan notebook

# 3. (Opsional) Jalankan ulang notebook untuk membuat ulang model
jupyter notebook notebook.ipynb
```

> Versi `scikit-learn`, `xgboost`, dan `lightgbm` di `requirements.txt` sengaja dikunci. File `model.joblib` dibuat dengan versi tersebut. Versi lain dapat membuat model gagal dimuat.

## Business Dashboard

Dashboard membantu pimpinan dan tim akademik memantau risiko _dropout_ tanpa membaca kode. Dashboard memakai file `dashboard_data.csv`. File ini berisi label yang mudah dibaca, kelompok usia, rasio kelulusan, **skor risiko dropout**, dan **tingkat risiko** (Rendah, Sedang, Tinggi) untuk setiap mahasiswa.

Skor risiko berasal dari prediksi _out-of-fold_. Model menilai tiap mahasiswa tanpa pernah melihat data mahasiswa itu. Dengan cara ini skor tidak bocor. Pada skor ini, kelompok risiko **Tinggi benar-benar dropout 90,8%**, **Sedang 46,3%**, dan **Rendah 9,3%**.

**Link dashboard:** _(tempel link Looker Studio / Tableau Public Anda di sini. Pastikan akses "siapa saja yang memiliki link".)_

### Metrik utama yang dipantau

| Metrik                                 | Definisi                                               | Mengapa penting                                      |
| -------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| Tingkat dropout                        | Mahasiswa `Dropout` / total mahasiswa                  | Indikator utama. Nilai dasar: 32,1%.                 |
| Tingkat kelulusan                      | Mahasiswa `Graduate` / total mahasiswa                 | Pasangan dari tingkat dropout. Nilai dasar: 49,9%.   |
| Mahasiswa risiko tinggi                | Jumlah mahasiswa dengan `Risk_level = Tinggi`          | Daftar prioritas bimbingan.                          |
| Tingkat menunggak                      | Mahasiswa dengan `Tuition_fees_up_to_date = 0` / total | Penunggak dropout sekitar 87%.                       |
| Rasio kelulusan MK semester 2          | MK lulus / MK diambil                                  | Fitur terpenting model.                              |
| Mahasiswa tanpa MK lulus di semester 2 | Jumlah dan persentase                                  | Sekitar 51% mahasiswa dropout berada di kondisi ini. |
| Cakupan beasiswa                       | Penerima beasiswa / total                              | Penerima dropout sekitar 12%, non-penerima 39%.      |
| Dropout per program studi              | Dropout / total per `Course_name`                      | Menemukan program yang bermasalah.                   |

### layout dashboard (Looker Studio)

Gunakan 4 halaman (tab). Beri filter global di bagian atas: **Program studi**, **Kelompok usia**, **Gender**, **Waktu kuliah**, dan **Status**.

**Halaman 1: Ringkasan eksekutif**

| Komponen                  | Jenis chart                                           | Field                                                     |
| ------------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| 4 kartu angka (KPI)       | Scorecard                                             | Total mahasiswa, % dropout, % lulus, jumlah risiko tinggi |
| Komposisi status          | Donut chart                                           | `Status`                                                  |
| Dropout per program studi | Bar chart horizontal, urut turun, garis rata-rata 32% | `Course_name` x % dropout                                 |
| Dropout per kelompok usia | Bar chart                                             | `Age_group` x % dropout                                   |

**Halaman 2: Keuangan**

| Komponen                             | Jenis chart                    | Field                                                 |
| ------------------------------------ | ------------------------------ | ----------------------------------------------------- |
| Status dropout menurut pembayaran    | Stacked bar 100%               | `Tuition_fees_up_to_date_label` x `Status`            |
| Dropout menurut beasiswa dan debitur | Stacked bar 100% (dua chart)   | `Scholarship_holder_label`, `Debtor_label`            |
| Pembayaran x beasiswa                | Heatmap / pivot table berwarna | Baris: pembayaran. Kolom: beasiswa. Nilai: % dropout. |

**Halaman 3: Akademik**

| Komponen                            | Jenis chart                      | Field                                                                    |
| ----------------------------------- | -------------------------------- | ------------------------------------------------------------------------ |
| MK lulus semester 1 vs semester 2   | Scatter plot berwarna per status | `Curricular_units_1st_sem_approved`, `Curricular_units_2nd_sem_approved` |
| Distribusi nilai semester           | Box plot per status              | `Curricular_units_1st_sem_grade`, `Curricular_units_2nd_sem_grade`       |
| Rata-rata MK lulus                  | Bar chart berkelompok            | Semester x `Status`                                                      |
| Persentase 0 MK lulus di semester 2 | Bar chart                        | `Curricular_units_2nd_sem_approved = 0` per `Status`                     |

**Halaman 4: Daftar pantau (early warning)**

| Komponen                       | Jenis chart                                                         | Field                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Distribusi tingkat risiko      | Bar chart / donut                                                   | `Risk_level`                                                                                                           |
| Daftar mahasiswa berisiko      | Tabel dengan pewarnaan kondisional, urut `Risk_score_dropout` turun | `Student_ID`, `Course_name`, `Risk_score_dropout`, `Risk_level`, `Tuition_fees_up_to_date_label`, `Approval_rate_sem2` |
| Risiko per program             | Heatmap                                                             | `Course_name` x `Risk_level`                                                                                           |
| Skor risiko vs rasio kelulusan | Scatter plot                                                        | `Approval_rate_sem2` x `Risk_score_dropout`                                                                            |

**Langkah singkat di Looker Studio:**

1. Buka Looker Studio, pilih **Create > Data source > File Upload**, lalu unggah `dashboard_data.csv`.
2. Buat _calculated field_ `Dropout_flag` = `CASE WHEN Status = "Dropout" THEN 1 ELSE 0 END`. Rata-rata field ini menghasilkan % dropout.
3. Buat 4 halaman sesuai tabel di atas. Pasang filter global.
4. Beri warna konsisten: merah untuk Dropout, kuning untuk Enrolled, teal untuk Graduate.
5. Atur akses berbagi agar reviewer dapat membuka dashboard, lalu simpan screenshot sebagai `username_dicoding-dashboard.png`.

## Menjalankan Sistem Machine Learning

Prototype memakai Streamlit. Aplikasi memuat model `model/model.joblib`, menerima data mahasiswa, lalu menampilkan prediksi `Dropout`, `Active/Enrolled`, atau `Graduate`.

**Jalankan secara lokal:**

```bash
# 1. Pasang dependensi (lihat bagian Persiapan)
pip install -r requirements.txt

# 2. Jalankan aplikasi dari folder proyek
streamlit run app.py atau python -m streamlit run app.py

```

Streamlit membuka aplikasi di `http://localhost:8501`.

**Cara memakai aplikasi:**

1. Isi data mahasiswa di **sidebar** (profil, keuangan, semester 1, semester 2). Atau pilih **Muat contoh profil** untuk mengisi form dengan cepat.
2. Lihat hasil di tab **Prediksi satu mahasiswa**. Aplikasi menampilkan status prediksi, peluang tiap kelas, tingkat risiko, dan saran tindak lanjut.
3. Untuk banyak mahasiswa, buka tab **Prediksi massal (CSV)**. Unggah file dengan kolom seperti `data.csv`, lalu unduh hasilnya.
4. Aplikasi memberi peringatan bila keyakinan model di bawah 70%. Tinjau kasus seperti itu secara manual.

**Deploy ke Streamlit Community Cloud:**

1. Unggah folder proyek ke repositori GitHub. Pastikan `app.py`, `requirements.txt`, dan `model/model.joblib` ikut terunggah.
2. Buka [share.streamlit.io](https://share.streamlit.io), lalu pilih **Create app**.
3. Pilih repositori, branch, dan file utama `app.py`.
4. Pada **Advanced settings**, pilih Python 3.12 (minimal 3.11).
5. Klik **Deploy**. Tunggu hingga aplikasi aktif.

**Link prototype:** _(tempel link Streamlit Community Cloud Anda di sini)_

## Conclusion

**Kondisi data.** Sebanyak 32,1% mahasiswa dropout, 49,9% lulus, dan 17,9% masih aktif. Data bersih tanpa nilai kosong atau duplikat.

**Faktor yang paling berkaitan dengan dropout.**

- **Kelulusan mata kuliah.** Rata-rata mata kuliah lulus semester 1 adalah 2,6 (Dropout), 4,3 (Enrolled), dan 6,2 (Graduate). Sekitar 51% mahasiswa dropout tidak lulus satu pun mata kuliah di semester 2. Pada lulusan angkanya hanya 3%.
- **Keuangan.** Mahasiswa yang menunggak biaya kuliah dropout sekitar 87%. Yang membayar tepat waktu dropout sekitar 25%. Debitur dropout sekitar 62%.
- **Beasiswa.** Penerima beasiswa dropout sekitar 12%. Non-penerima sekitar 39%.
- **Usia.** Risiko naik tajam setelah usia 22 tahun: sekitar 43% pada usia 23-25 dan mendekati 60% pada usia 26-30.
- **Gender.** Mahasiswa laki-laki dropout sekitar 45%. Mahasiswa perempuan sekitar 25%.
- **Program studi.** Keperawatan hanya sekitar 15%. Teknik Informatika, Equinikultur, dan Manajemen kelas malam berada di atas 50%.
- **Faktor yang lemah.** Nilai masuk, kewarganegaraan, status internasional, kebutuhan khusus, dan inflasi hampir tidak membedakan status.

**Performa model.** XGBoost terpilih karena macro F1 validasi silangnya tertinggi (0,739). Hasil pada data uji (885 mahasiswa):

| Metrik                  | Nilai |
| ----------------------- | ----- |
| Accuracy                | 73,7% |
| Macro F1                | 0,694 |
| ROC-AUC (one-vs-rest)   | 0,889 |
| Precision kelas Dropout | 0,81  |
| Recall kelas Dropout    | 0,69  |

Model berkinerja baik untuk `Dropout` dan `Graduate`. Model lemah pada kelas `Enrolled` (F1 sekitar 0,50). Status ini berada di antara dua kelas lain dan belum final. Keempat model berkinerja mirip. Selisihnya masih dalam rentang variasi validasi silang.

**Fitur terpenting model.** `approval_rate_sem2` (rasio kelulusan semester 2) dan `Tuition_fees_up_to_date` (pembayaran tepat waktu). Fitur lain berkontribusi jauh lebih kecil.

**Seberapa dini bisa mendeteksi?**

| Titik waktu      | Accuracy | Macro F1 | ROC-AUC |
| ---------------- | -------- | -------- | ------- |
| Saat pendaftaran | 58,4%    | 0,549    | 0,773   |
| Akhir semester 1 | 71,1%    | 0,667    | 0,866   |
| Akhir semester 2 | 73,7%    | 0,694    | 0,889   |

Data pendaftaran saja belum cukup. Setelah semester 1, model sudah mendekati kinerja penuh. Maka institusi sebaiknya menjalankan peringatan dini **tepat setelah semester 1**.

**Keyakinan model.** Prediksi dengan keyakinan di atas 0,9 benar 97,4%. Prediksi dengan keyakinan di bawah 0,7 benar hanya 41-51%. Staf perlu meninjau kasus yang tidak yakin secara manual.

**Keterbatasan.**

- Data berasal dari satu institusi. Uji ulang model sebelum memakainya pada kohort baru.
- Hasil menunjukkan hubungan, bukan sebab-akibat.
- Aplikasi Streamlit memakai model penuh (semester 1 dan 2). Model khusus semester 1 baru dievaluasi di notebook.

## Rekomendasi Action Items

1. **Bangun peringatan dini keuangan.** Tandai mahasiswa yang menunggak atau berstatus debitur sejak bulan pertama. Tawarkan cicilan, penjadwalan ulang, atau keringanan. Sasarannya kelompok dengan dropout sekitar 87% (menunggak) dan 62% (debitur). Pantau setiap bulan dengan metrik tingkat menunggak di dashboard.
2. **Perluas beasiswa dengan target tepat.** Penerima beasiswa dropout 12% dibanding 39% pada non-penerima. Prioritaskan mahasiswa yang tidak berbeasiswa, tetapi menunggak atau berstatus debitur. Beasiswa saja tidak cukup untuk penunggak, jadi gabungkan dengan konseling keuangan.
3. **Jalankan program bimbingan akademik setelah semester 1.** Jalankan model pada akhir semester 1. Beri mentor, kelas remedial, atau tutor sebaya kepada mahasiswa berisiko Tinggi dan Sedang. Beri perhatian ekstra bila rasio kelulusan di bawah 50%. Beri perhatian segera bila tidak ada mata kuliah lulus.
4. **Dukung mahasiswa dewasa dan kelas malam.** Dropout melonjak setelah usia 22 tahun. Tawarkan jadwal fleksibel, kelas hibrida, dan layanan konsultasi di luar jam kerja. Kelas malam dropout sekitar 43% dibanding 31% pada kelas siang.
5. **Evaluasi program studi berisiko.** Periksa kurikulum, beban SKS, dan dukungan dosen pada Teknik Informatika, Equinikultur, dan Manajemen kelas malam. Pelajari praktik baik dari Keperawatan (dropout sekitar 15%).
6. **Jadikan skor risiko bagian dari proses rutin.** Hasilkan daftar risiko tiap akhir semester dari aplikasi atau `dashboard_data.csv`. Mulai dari risiko Tinggi (90,8% benar-benar dropout pada data ini). Tinjau manual prediksi dengan keyakinan di bawah 70%.
7. **Perkaya data dan perbarui model.** Tambahkan data kehadiran, aktivitas e-learning, status bekerja, dan hasil survei kepuasan. Data ini dapat memperbaiki prediksi kelas `Enrolled`. Latih ulang model setiap tahun dan ukur dampak intervensi (turunnya tingkat dropout pada kelompok yang dibimbing).

**Contoh target yang dapat dipantau di dashboard** _(usulan, sesuaikan dengan kebijakan institusi)_: tingkat dropout, jumlah mahasiswa risiko tinggi yang sudah dihubungi, tingkat menunggak, dan rasio kelulusan mata kuliah semester 1.

## Acknowledgements

Realinho, V., Vieira Martins, M., Machado, J., dan Baptista, L. (2021). _Predict students' dropout and academic success._ UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
