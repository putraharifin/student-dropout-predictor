# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institut

Proyek ini membangun sistem deteksi dini mahasiswa yang berisiko *dropout*. Sistem terdiri dari analisis data, model machine learning, prototype Streamlit, dan rancangan dashboard monitoring.

- **Nama:** Muhammad Putra Harifin Pane
- **Email:** putraharifin@gmail.com
- **Id Dicoding:** putraharifin25

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang berdiri sejak tahun 2000. Institusi ini telah meluluskan banyak alumni dengan reputasi baik. Namun, seiring berjalannya waktu, institusi menghadapi tantangan serius terkait banyaknya mahasiswa yang tidak menyelesaikan studinya atau memutuskan untuk _dropout_.

Fenomena tingkat _dropout_ yang tinggi ini membawa dampak negatif yang signifikan bagi berbagai pihak:
* **Bagi Mahasiswa:** Kehilangan waktu, biaya investasi pendidikan, serta kesempatan karir masa depan yang lebih baik.
* **Bagi Institusi:** Kehilangan potensi pendapatan dari biaya kuliah (SPP) secara berkelanjutan, menurunnya efisiensi alokasi kapasitas kelas, serta berisiko menurunkan reputasi atau nilai akreditasi institusi di mata publik.

Untuk mengatasi masalah ini, Jaya Jaya Institut berniat beralih dari pendekatan reaktif menjadi proaktif. Institusi ingin memanfaatkan data historis akademik dan profil mahasiswa untuk mendeteksi tanda-tanda kerentanan mahasiswa **sedini mungkin**. Dengan deteksi dini ini, pihak manajemen dan tim akademik dapat memberikan intervensi berupa bimbingan khusus atau konseling sebelum mahasiswa benar-benar mengambil keputusan untuk mundur.


## Permasalahan Bisnis

Tingginya angka mahasiswa yang tidak menyelesaikan studi di Jaya Jaya Institut menjadi ancaman nyata bagi stabilitas operasional, finansial, dan reputasi jangka panjang institusi. Masalah utama yang dihadapi saat ini adalah mekanisme penanganan mahasiswa yang masih bersifat reaktif. Pihak manajemen dan tim penasihat akademik umumnya baru mengetahui mahasiswa keluar setelah proses *dropout* resmi terdaftar pada administrasi, sehingga momentum untuk memberikan bantuan akademik atau konseling psikologis sudah terlambat.

Kondisi reaktif ini disebabkan oleh ketiadaan sistem pemantauan terpusat dan alat deteksi dini yang mampu mengidentifikasi mahasiswa rentan secara *real-time*. Institusi belum memiliki visibilitas yang jelas mengenai bagaimana faktor demografi (seperti usia saat mendaftar atau latar belakang keluarga), faktor finansial (seperti status beasiswa dan kelancaran pembayaran SPP), serta performa akademik pada semester-semester awal saling berinteraksi dalam memengaruhi keputusan mahasiswa untuk berhenti kuliah. Tanpa adanya informasi ini, alokasi sumber daya untuk program bimbingan mahasiswa menjadi tidak terarah dan kurang tepat sasaran.

Oleh karena itu, proyek data science ini bertujuan untuk menjawab permasalahan tersebut dengan membangun sistem prediksi otomatis berbasis machine learning beserta dashboard monitoring. Sistem ini dirancang untuk menyaring rekam jejak mahasiswa dan mengidentifikasi mereka yang berada dalam zona berisiko tinggi (*high-risk*) sejak awal masa studi mereka. Melalui solusi digital ini, Jaya Jaya Institut dapat mengimplementasikan program intervensi yang terukur, tepat waktu, dan tepat sasaran demi menekan laju angka *dropout* mahasiswa.


## Cakupan Proyek

**Yang dikerjakan:**
- Analisis data eksploratif (EDA) mendalam: distribusi, outlier, korelasi, uji statistik (Chi-square dan Kruskal-Wallis), serta analisis per faktor keuangan, akademik, demografi, dan program studi.
- Feature engineering dari kinerja akademik (rasio kelulusan, tren nilai).
- **Filtering data.** Hanya mahasiswa berstatus `Dropout` dan `Graduate` (3.630 mahasiswa) yang dipakai untuk melatih model. Status `Enrolled` (794 mahasiswa) belum punya hasil akhir. Jika ikut dilatih, target menjadi tidak jelas. Karena itu kita memisahkannya.
- **Target biner:** `1 = Dropout`, `0 = Graduate`.
- Model klasifikasi biner. Model yang dibandingkan: Logistic Regression, Random Forest, XGBoost, dan LightGBM.
- **Prediksi data Enrolled.** Setelah model jadi, kita memakainya untuk menilai potensi dropout mahasiswa `Enrolled`.
- Analisis ambang keputusan dan skenario deteksi dini (saat pendaftaran, akhir semester 1, akhir semester 2).
- Prototype aplikasi Streamlit untuk prediksi satu mahasiswa dan prediksi massal lewat CSV.
- Data siap pakai untuk dashboard (`dashboard_data.csv`) dan rancangan layout dashboard.
- Rekomendasi *action items*.

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

**Setup environment:** Python yang digunakan versi 3.11.

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

Dashboard membantu pimpinan dan tim akademik memantau risiko *dropout* tanpa membaca kode. Dashboard memakai file `dashboard_data.csv`. File ini berisi label yang mudah dibaca, kelompok usia, rasio kelulusan, **skor risiko dropout**, dan **tingkat risiko** (Rendah, Sedang, Tinggi) untuk setiap mahasiswa.

Asal skor risiko dijelaskan di kolom `Score_source`:
- **Mahasiswa Dropout dan Graduate:** skor *out-of-fold*. Model menilai tiap mahasiswa tanpa pernah melihat data mahasiswa itu, sehingga skor tidak bocor. Pada mahasiswa berlabel, kelompok risiko **Tinggi benar-benar dropout 93,5%**, **Sedang 37,5%**, dan **Rendah 7,7%**.
- **Mahasiswa Enrolled:** skor dari model final. Ini prediksi potensi dropout untuk masa mendatang. Dari 794 mahasiswa `Enrolled`, 370 masuk risiko Tinggi, 83 Sedang, dan 341 Rendah.

**Link dashboard:** __https://datastudio.google.com/reporting/6d7ba714-3d61-4b09-997f-fd2b0ae6e8f8_

### Metrik utama yang dipantau

| Metrik | Definisi | Mengapa penting |
|--------|----------|-----------------|
| Tingkat dropout | Mahasiswa `Dropout` / total mahasiswa | Indikator utama. Nilai dasar: 32,1%. |
| Tingkat kelulusan | Mahasiswa `Graduate` / total mahasiswa | Pasangan dari tingkat dropout. Nilai dasar: 49,9%. |
| Mahasiswa risiko tinggi | Jumlah mahasiswa dengan `Risk_level = Tinggi` | Daftar prioritas bimbingan. |
| Mahasiswa Enrolled berisiko tinggi | `Status = Enrolled` dan `Risk_level = Tinggi` | Mahasiswa aktif yang masih bisa dibantu sekarang. |
| Tingkat menunggak | Mahasiswa dengan `Tuition_fees_up_to_date = 0` / total | Penunggak dropout sekitar 87%. |
| Rasio kelulusan MK semester 2 | MK lulus / MK diambil | Fitur terpenting model. |
| Mahasiswa tanpa MK lulus di semester 2 | Jumlah dan persentase | Sekitar 51% mahasiswa dropout berada di kondisi ini. |
| Cakupan beasiswa | Penerima beasiswa / total | Penerima dropout sekitar 12%, non-penerima 39%. |
| Dropout per program studi | Dropout / total per `Course_name` | Menemukan program yang bermasalah. |


### Layout Dashboard (Looker Studio)

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

| Komponen                            | Jenis chart               | Field                                                                    |
| ----------------------------------- | ------------------------- | ------------------------------------------------------------------------ |
| MK lulus semester 1 vs semester 2   | Scatter plot        | `Curricular_units_1st_sem_approved`, `Curricular_units_2nd_sem_approved` |
| Distribusi nilai semester           | Line Chart (Grafik Garis)       | `Curricular_units_1st_sem_grade`, `Curricular_units_2nd_sem_grade`       |
| Rata-rata MK lulus                  | Bar chart berkelompok     | Semester x `Status`                                                      |
| Persentase 0 MK lulus di semester 2 | Bar chart                 | `Curricular_units_2nd_sem_approved = 0` per `Status`                     |

**Halaman 4: Daftar pantau (early warning)**

| Komponen                       | Jenis chart                                                         | Field                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Distribusi tingkat risiko      |  donut chart                                                   | `Risk_level`                                                                                                           |
| Daftar mahasiswa berisiko      | Tabel dengan pewarnaan kondisional, urut `Risk_score_dropout` turun | `Student_ID`, `Course_name`, `Risk_score_dropout`, `Risk_level`, `Tuition_fees_up_to_date_label`, `Approval_rate_sem2` |
| Risiko per program             | Heatmap                                                             | `Course_name` x `Risk_level`                                                                                           |
| Skor risiko vs rasio kelulusan | Scatter plot                                                        | `Approval_rate_sem2` x `Risk_score_dropout`                                                                            |

**Langkah singkat di Looker Studio:**

1. Buka Looker Studio, pilih **Create > Data source > File Upload**, lalu unggah `dashboard_data.csv`.
2. Buat _calculated field_ `Dropout_flag` = `CASE WHEN Status = "Dropout" THEN 1 ELSE 0 END`. Rata-rata field ini menghasilkan % dropout.
3. Buat 4 halaman sesuai tabel di atas. Pasang filter global.
4. Beri warna konsisten: merah untuk Dropout, kuning untuk Enrolled, teal untuk Graduate.
5. Atur akses berbagi agar reviewer dapat membuka dashboard, lalu simpan screenshot sebagai `putraharifin25-dashboard.png`.
 


## Menjalankan Sistem Machine Learning

Prototype memakai Streamlit. Aplikasi memuat model `model/model.joblib`, menerima data mahasiswa, lalu menampilkan prediksi **Dropout** atau **Graduate** beserta peluangnya. Mahasiswa berstatus `Enrolled` dapat dinilai juga. Hasilnya adalah potensi di masa mendatang.

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
2. Lihat hasil di tab **Prediksi satu mahasiswa**. Aplikasi menampilkan prediksi (Dropout atau Graduate), peluang Dropout, tingkat risiko, dan saran tindak lanjut.
3. Untuk banyak mahasiswa, buka tab **Prediksi massal (CSV)**. Unggah file dengan kolom seperti `data.csv` (boleh memuat mahasiswa `Enrolled`), lalu unduh hasilnya.
4. Aplikasi memberi peringatan bila keyakinan model di bawah 70%. Tinjau kasus seperti itu secara manual.
5. Prediksi dropout memakai ambang 0,5. Peluang di atas ambang berarti `Dropout`.

**Deploy ke Streamlit Community Cloud:**

1. Unggah folder proyek ke repositori GitHub. Pastikan `app.py`, `requirements.txt`, dan `model/model.joblib` ikut terunggah.
2. Buka [share.streamlit.io](https://share.streamlit.io), lalu pilih **Create app**.
3. Pilih repositori, branch, dan file utama `app.py`.
4. Pada **Advanced settings**, pilih Python 3.11.
5. Klik **Deploy**. Tunggu hingga aplikasi aktif.

**Link prototype:** _https://student-dropout-predictor-a5w4awroq9fnjtr8imm4xn.streamlit.app/_

## Conclusion

**Kondisi data.** Dataset berisi 4.424 mahasiswa: 49,9% lulus, 32,1% dropout, dan 17,9% masih aktif (`Enrolled`). Data bersih tanpa nilai kosong atau duplikat.

**Penyiapan data untuk model.** Mahasiswa `Enrolled` belum punya hasil akhir, sehingga kita mengeluarkannya dari pelatihan. Data modeling berisi 3.630 mahasiswa: 2.209 `Graduate` (60,9%) dan 1.421 `Dropout` (39,1%). Target dibuat biner (1 = Dropout, 0 = Graduate). Data dibagi 80:20 dengan `stratify`: 2.904 data latih dan 726 data uji. Sebanyak 794 mahasiswa `Enrolled` disimpan sebagai data prediksi.

**Faktor yang paling berkaitan dengan dropout.**
- **Kelulusan mata kuliah.** Rata-rata mata kuliah lulus semester 1 adalah 2,6 (Dropout), 4,3 (Enrolled), dan 6,2 (Graduate). Sekitar 51% mahasiswa dropout tidak lulus satu pun mata kuliah di semester 2. Pada lulusan angkanya hanya 3%.
- **Keuangan.** Mahasiswa yang menunggak biaya kuliah dropout sekitar 87%. Yang membayar tepat waktu dropout sekitar 25%. Debitur dropout sekitar 62%.
- **Beasiswa.** Penerima beasiswa dropout sekitar 12%. Non-penerima sekitar 39%.
- **Usia.** Risiko naik tajam setelah usia 22 tahun: sekitar 43% pada usia 23-25 dan mendekati 60% pada usia 26-30.
- **Gender.** Mahasiswa laki-laki dropout sekitar 45%. Mahasiswa perempuan sekitar 25%.
- **Program studi.** Keperawatan hanya sekitar 15%. Teknik Informatika, Equinikultur, dan Manajemen kelas malam berada di atas 50%.
- **Faktor yang lemah.** Nilai masuk, kewarganegaraan, status internasional, kebutuhan khusus, dan inflasi hampir tidak membedakan status.

(Angka EDA di atas dihitung pada seluruh mahasiswa, termasuk `Enrolled`, agar gambaran institusi utuh.)

**Performa model.** LightGBM terpilih karena F1 Dropout validasi silangnya tertinggi (0,874). Hasil pada data uji (726 mahasiswa):

| Metrik | Nilai |
|--------|-------|
| Accuracy | 93,1% |
| Precision kelas Dropout | 0,903 |
| Recall kelas Dropout | 0,923 |
| F1 kelas Dropout | 0,913 |
| ROC-AUC | 0,973 |

Dari 726 mahasiswa uji, model melewatkan 22 calon dropout dan menandai 28 calon lulusan secara keliru. Keempat algoritma berkinerja tinggi dan mirip. Logistic Regression baseline juga kuat (akurasi 92,1%, ROC-AUC 0,976). Selisihnya masih dalam rentang variasi validasi silang.

**Fitur terpenting model.** `approval_rate_sem2` (rasio kelulusan semester 2) dan `Tuition_fees_up_to_date` (pembayaran tepat waktu). Fitur lain berkontribusi jauh lebih kecil.

**Ambang keputusan.** Ambang 0,5 memberi keseimbangan yang baik. Ambang 0,3 menaikkan recall ke 0,937, tetapi presisi turun ke 0,847. Ambang 0,7 menaikkan presisi ke 0,945, tetapi recall turun ke 0,908. Institusi dapat memilih sesuai kapasitas bimbingan.

**Seberapa dini bisa mendeteksi?**

| Titik waktu | Accuracy | Recall Dropout | F1 Dropout | ROC-AUC |
|-------------|----------|----------------|------------|---------|
| Saat pendaftaran | 76,6% | 0,697 | 0,700 | 0,844 |
| Akhir semester 1 | 88,8% | 0,877 | 0,860 | 0,950 |
| Akhir semester 2 | 93,1% | 0,923 | 0,913 | 0,973 |

Data pendaftaran saja belum cukup. Setelah semester 1, model sudah mendekati kinerja penuh. Maka institusi sebaiknya menjalankan peringatan dini **tepat setelah semester 1**.

**Keyakinan model.** Prediksi dengan keyakinan di atas 0,9 benar 96,8% (622 dari 726 kasus). Prediksi dengan keyakinan 0,5-0,7 benar hanya sekitar 56-61%. Staf perlu meninjau kasus yang tidak yakin secara manual.

**Prediksi mahasiswa Enrolled.** Model menilai 794 mahasiswa `Enrolled`: 398 diprediksi berpotensi `Dropout` dan 396 `Graduate`. Sebanyak 370 mahasiswa masuk risiko Tinggi. Mahasiswa `Enrolled` yang menunggak uang kuliah punya rata-rata peluang dropout 0,84, dibanding 0,48 pada yang membayar tepat waktu. Ini hasil model, bukan fakta. Model tidak bisa divalidasi pada kelompok ini karena labelnya belum ada.

**Keterbatasan.**
- Data berasal dari satu institusi. Uji ulang model sebelum memakainya pada kohort baru.
- Hasil menunjukkan hubungan, bukan sebab-akibat.
- Mahasiswa `Enrolled` mungkin berbeda dari populasi pelatihan, karena mereka masih kuliah di akhir masa studi normal. Anggap prediksi mereka sebagai sinyal untuk ditinjau, bukan vonis.
- Aplikasi Streamlit memakai model penuh (semester 1 dan 2). Model khusus semester 1 baru dievaluasi di notebook.

## Rekomendasi Action Items

1. **Bangun peringatan dini keuangan.** Tandai mahasiswa yang menunggak atau berstatus debitur sejak bulan pertama. Tawarkan cicilan, penjadwalan ulang, atau keringanan. Sasarannya kelompok dengan dropout sekitar 87% (menunggak) dan 62% (debitur). Pantau setiap bulan dengan metrik tingkat menunggak di dashboard.
2. **Perluas beasiswa dengan target tepat.** Penerima beasiswa dropout 12% dibanding 39% pada non-penerima. Prioritaskan mahasiswa yang tidak berbeasiswa, tetapi menunggak atau berstatus debitur. Beasiswa saja tidak cukup untuk penunggak, jadi gabungkan dengan konseling keuangan.
3. **Jalankan program bimbingan akademik setelah semester 1.** Jalankan model pada akhir semester 1. Beri mentor, kelas remedial, atau tutor sebaya kepada mahasiswa berisiko Tinggi dan Sedang. Beri perhatian ekstra bila rasio kelulusan di bawah 50%. Beri perhatian segera bila tidak ada mata kuliah lulus.
4. **Dukung mahasiswa dewasa dan kelas malam.** Dropout melonjak setelah usia 22 tahun. Tawarkan jadwal fleksibel, kelas hibrida, dan layanan konsultasi di luar jam kerja. Kelas malam dropout sekitar 43% dibanding 31% pada kelas siang.
5. **Evaluasi program studi berisiko.** Periksa kurikulum, beban SKS, dan dukungan dosen pada Teknik Informatika, Equinikultur, dan Manajemen kelas malam. Pelajari praktik baik dari Keperawatan (dropout sekitar 15%).
6. **Jadikan skor risiko bagian dari proses rutin.** Hasilkan daftar risiko tiap akhir semester dari aplikasi atau `dashboard_data.csv`. Mulai dari risiko Tinggi (93,5% benar-benar dropout pada data berlabel). Tinjau manual prediksi dengan keyakinan di bawah 70%.
7. **Perkaya data dan perbarui model.** Tambahkan data kehadiran, aktivitas e-learning, status bekerja, dan hasil survei kepuasan. Data ini dapat memperkuat prediksi untuk mahasiswa yang polanya tidak jelas. Setelah status akhir mahasiswa `Enrolled` diketahui, tambahkan mereka ke data latih. Latih ulang model setiap tahun dan ukur dampak intervensi (turunnya tingkat dropout pada kelompok yang dibimbing).

8. **Prioritaskan mahasiswa `Enrolled` berisiko tinggi sekarang.** Sebanyak 370 mahasiswa aktif masuk risiko Tinggi. Mereka masih bisa dibantu sebelum status akhir terbentuk. Hubungi dulu yang menunggak biaya kuliah atau yang tidak lulus mata kuliah di semester terakhir. Catat hasil intervensi, lalu bandingkan dengan prediksi model.

**Contoh target yang dapat dipantau di dashboard** *(usulan, sesuaikan dengan kebijakan institusi)*: tingkat dropout, jumlah mahasiswa risiko tinggi yang sudah dihubungi, tingkat menunggak, dan rasio kelulusan mata kuliah semester 1.

## Acknowledgements

Realinho, V., Vieira Martins, M., Machado, J., dan Baptista, L. (2021). *Predict students' dropout and academic success.* UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
