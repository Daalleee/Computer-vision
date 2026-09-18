# Laporan Praktikum Minggu 3
## Deteksi Tepi & Ekstraksi Fitur Citra

**Nama:** Leonardus Dale Masan
**NIM:** 235314071
**Mata Kuliah:** Computer Vision

---

# 1. Deteksi Tepi (Sobel vs Canny)

Pada praktikum ini digunakan sebuah citra sebagai input untuk melakukan deteksi tepi menggunakan metode Sobel dan Canny.

Sobel digunakan untuk memperoleh magnitudo gradien citra berdasarkan perubahan intensitas pada arah horizontal dan vertikal.

Canny digunakan dengan tiga pasangan threshold, yaitu threshold longgar, sedang, dan ketat.

### Hasil Canny

| Pengaturan | Threshold | Jumlah Piksel Tepi | Persentase |
|---|---:|---:|---:|
| Longgar | 30-90 | 584886 | 7.80% |
| Sedang | 100-200 | 112070 | 1.50% |
| Ketat | 150-250 | 73474 | 0.98% |


### Analisis

Pada threshold yang lebih rendah, Canny cenderung mendeteksi lebih banyak tepi karena lebih banyak perubahan intensitas yang dianggap sebagai kandidat tepi.

Pada threshold yang lebih tinggi, deteksi menjadi lebih selektif sehingga hanya tepi yang memiliki kekuatan lebih tinggi yang dipertahankan.

Berdasarkan hasil percobaan, pengaturan threshold yang digunakan menghasilkan perbedaan jumlah piksel tepi seperti pada tabel di atas.

**Threshold yang paling sesuai untuk citra ini:** ISI ANALISIS SENDIRI.

**Alasan:** ISI ANALISIS SENDIRI.

Lampiran:
`tugas1_sobel_vs_canny.png`

---

# 2. Perbandingan SIFT vs ORB

Benchmark dilakukan sebanyak **20 kali pengulangan** pada citra yang sama.

| Detektor | Jumlah Keypoint | Waktu Eksekusi Rata-rata |
|---|---:|---:|
| SIFT | 8880 | 1396.79 ms |
| ORB | 500 | 62.33 ms |

Berdasarkan hasil benchmark, waktu eksekusi SIFT adalah **1396.79 ms**, sedangkan ORB adalah **62.33 ms**.

Pada pengujian ini, ORB sekitar **22.41 kali** lebih cepat daripada SIFT.

Jumlah keypoint SIFT adalah **8880**, sedangkan ORB adalah **500**.

### Pertanyaan Reflektif

**Kapan sebaiknya memilih ORB dibanding SIFT?**

ISI JAWABAN SENDIRI.

Pertimbangkan:
- kebutuhan real-time;
- kecepatan pemrosesan;
- keterbatasan CPU;
- perangkat edge;
- kebutuhan jumlah/keypoint;
- kebutuhan aplikasi.

Lampiran:
- `tugas2_keypoints_sift_vs_orb.png`
- `tugas2_benchmark_sift_vs_orb.png`

---

# 3. Kesimpulan

ISI KESIMPULAN SENDIRI.

Kesimpulan sebaiknya membahas:
1. hasil Sobel dan Canny;
2. pengaruh threshold Canny;
3. perbandingan jumlah keypoint SIFT dan ORB;
4. perbandingan waktu eksekusi SIFT dan ORB;
5. pertimbangan pemilihan metode berdasarkan kebutuhan aplikasi.

---

# File Hasil

- `citra_asli_grayscale.png`
- `tugas1_sobel_vs_canny.png`
- `tugas2_keypoints_sift_vs_orb.png`
- `tugas2_benchmark_sift_vs_orb.png`
- `laporan_praktikum_minggu3.md`
