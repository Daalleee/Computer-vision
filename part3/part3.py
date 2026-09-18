#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRAKTIKUM MINGGU 3
Deteksi Tepi & Ekstraksi Fitur Citra

Tugas:
1. Menerapkan Sobel dan Canny.
2. Membandingkan Canny dengan 3 pasangan threshold.
3. Membandingkan SIFT dan ORB berdasarkan:
   - jumlah keypoint
   - waktu eksekusi
4. Menghasilkan gambar hasil dan laporan.

Input:
- Menggunakan gambar milik mahasiswa.
- Letakkan gambar di folder yang sama dengan file Python.

Output:
- output/
    ├── citra_asli_grayscale.png
    ├── tugas1_sobel_vs_canny.png
    ├── tugas2_keypoints_sift_vs_orb.png
    ├── tugas2_benchmark_sift_vs_orb.png
    └── laporan_praktikum_minggu3.md
"""

# ============================================================
# IMPORT LIBRARY
# ============================================================

import cv2
import numpy as np
import matplotlib

# Agar bisa dijalankan tanpa GUI
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import time
from pathlib import Path


# ============================================================
# KONFIGURASI
# ============================================================

# ------------------------------------------------------------
# 1. GAMBAR INPUT
# ------------------------------------------------------------

IMAGE_PATH = "C:/Users/dalleno/Dokumen/komputer pelihat/part3//data//messi.jpg"


# ------------------------------------------------------------
# Nama folder untuk menyimpan hasil
# ------------------------------------------------------------

OUTPUT_DIR = "output"


# ------------------------------------------------------------
# Tiga setting threshold Canny
# ------------------------------------------------------------

CANNY_SETTINGS = [
    ("Longgar", 30, 90),
    ("Sedang", 100, 200),
    ("Ketat", 150, 250),
]


# ------------------------------------------------------------
# Jumlah pengulangan benchmark
# ------------------------------------------------------------

N_ULANGAN = 20


# ------------------------------------------------------------
# Maksimum fitur ORB
# ------------------------------------------------------------

JUMLAH_FITUR_MAKS = 500


# ============================================================
# FUNGSI 1
# MEMBUAT FOLDER OUTPUT
# ============================================================

def buat_folder_output():

    output_path = Path(OUTPUT_DIR)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    return output_path


# ============================================================
# FUNGSI 2
# MEMBACA GAMBAR
# ============================================================

def baca_gambar(path):

    print("=" * 60)
    print("MEMBACA GAMBAR")
    print("=" * 60)

    print(f"Path gambar : {path}")

    # Membaca gambar dalam format grayscale
    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    # Mengecek apakah gambar berhasil dibaca
    if image is None:

        raise FileNotFoundError(
            f"\nGambar tidak ditemukan!\n"
            f"Pastikan file '{path}' berada di folder yang sama "
            f"dengan program Python.\n"
        )

    print(f"Ukuran gambar : {image.shape[1]} x {image.shape[0]}")
    print("Gambar berhasil dibaca.")

    return image


# ============================================================
# FUNGSI 3
# MENYIMPAN GAMBAR ASLI
# ============================================================

def simpan_gambar_asli(image, output_path):

    file_path = output_path / "citra_asli_grayscale.png"

    cv2.imwrite(
        str(file_path),
        image
    )

    print(
        f"\n[OK] Citra grayscale disimpan:"
        f"\n    {file_path}"
    )


# ============================================================
# TUGAS 1
# SOBEL
# ============================================================

def hitung_sobel(image):

    # Gradien arah X
    gx = cv2.Sobel(
        image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    # Gradien arah Y
    gy = cv2.Sobel(
        image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    # Magnitudo gradien
    magnitude = cv2.magnitude(
        gx.astype(np.float32),
        gy.astype(np.float32)
    )

    # Normalisasi menjadi 0-255
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    magnitude = magnitude.astype(
        np.uint8
    )

    return magnitude


# ============================================================
# TUGAS 1
# SOBEL + CANNY
# ============================================================

def tugas1_deteksi_tepi(
    image,
    output_path
):

    print("\n")
    print("=" * 60)
    print("TUGAS 1 - DETEKSI TEPI")
    print("SOBEL VS CANNY")
    print("=" * 60)

    # --------------------------------------------------------
    # SOBEL
    # --------------------------------------------------------

    sobel = hitung_sobel(image)

    # --------------------------------------------------------
    # CANNY
    # --------------------------------------------------------

    hasil_canny = []

    for nama, threshold_low, threshold_high in CANNY_SETTINGS:

        hasil = cv2.Canny(
            image,
            threshold_low,
            threshold_high
        )

        jumlah_tepi = np.count_nonzero(
            hasil
        )

        total_pixel = hasil.size

        persentase = (
            jumlah_tepi /
            total_pixel
        ) * 100

        hasil_canny.append(
            (
                nama,
                threshold_low,
                threshold_high,
                hasil,
                jumlah_tepi,
                persentase
            )
        )

        print(
            f"\nCanny {nama}"
        )

        print(
            f"Threshold       : "
            f"{threshold_low} - {threshold_high}"
        )

        print(
            f"Jumlah tepi     : "
            f"{jumlah_tepi} piksel"
        )

        print(
            f"Persentase      : "
            f"{persentase:.2f}%"
        )

    # --------------------------------------------------------
    # MEMBUAT FIGURE
    # --------------------------------------------------------

    jumlah_panel = 2 + len(
        hasil_canny
    )

    fig, axes = plt.subplots(
        1,
        jumlah_panel,
        figsize=(
            4 * jumlah_panel,
            5
        )
    )

    # --------------------------------------------------------
    # CITRA ASLI
    # --------------------------------------------------------

    axes[0].imshow(
        image,
        cmap="gray"
    )

    axes[0].set_title(
        "Citra Asli"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # SOBEL
    # --------------------------------------------------------

    axes[1].imshow(
        sobel,
        cmap="gray"
    )

    axes[1].set_title(
        "Sobel |∇I|"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # CANNY
    # --------------------------------------------------------

    for index, data in enumerate(
        hasil_canny
    ):

        (
            nama,
            threshold_low,
            threshold_high,
            hasil,
            jumlah_tepi,
            persentase
        ) = data

        ax = axes[
            index + 2
        ]

        ax.imshow(
            hasil,
            cmap="gray"
        )

        ax.set_title(
            f"Canny {nama}\n"
            f"Threshold = "
            f"({threshold_low}, "
            f"{threshold_high})"
        )

        ax.set_xlabel(
            f"{jumlah_tepi} piksel\n"
            f"({persentase:.2f}%)"
        )

        ax.axis("off")

    # --------------------------------------------------------
    # JUDUL
    # --------------------------------------------------------

    fig.suptitle(
        "Tugas 1 - Sobel dan Canny",
        fontsize=15
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SIMPAN
    # --------------------------------------------------------

    output_file = (
        output_path /
        "tugas1_sobel_vs_canny.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n[OK] Hasil Tugas 1 disimpan:"
        f"\n    {output_file}"
    )

    return hasil_canny


# ============================================================
# TUGAS 2
# BENCHMARK SIFT / ORB
# ============================================================

def benchmark_detector(
    detector,
    image,
    n_ulangan
):

    # --------------------------------------------------------
    # WARM UP
    # --------------------------------------------------------

    keypoints, descriptors = (
        detector.detectAndCompute(
            image,
            None
        )
    )

    # --------------------------------------------------------
    # MULAI TIMER
    # --------------------------------------------------------

    waktu_mulai = time.perf_counter()

    for _ in range(
        n_ulangan
    ):

        detector.detectAndCompute(
            image,
            None
        )

    waktu_selesai = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # HITUNG WAKTU
    # --------------------------------------------------------

    total_waktu = (
        waktu_selesai -
        waktu_mulai
    )

    rata_rata_ms = (
        total_waktu /
        n_ulangan
    ) * 1000

    return (
        keypoints,
        rata_rata_ms
    )


# ============================================================
# TUGAS 2
# SIFT VS ORB
# ============================================================

def tugas2_sift_vs_orb(
    image,
    output_path
):

    print("\n")
    print("=" * 60)
    print("TUGAS 2 - SIFT VS ORB")
    print("=" * 60)

    # --------------------------------------------------------
    # MEMBUAT SIFT
    # --------------------------------------------------------

    sift = cv2.SIFT_create()

    # --------------------------------------------------------
    # MEMBUAT ORB
    # --------------------------------------------------------

    orb = cv2.ORB_create(
        nfeatures=JUMLAH_FITUR_MAKS
    )

    # --------------------------------------------------------
    # BENCHMARK SIFT
    # --------------------------------------------------------

    (
        keypoints_sift,
        waktu_sift
    ) = benchmark_detector(
        sift,
        image,
        N_ULANGAN
    )

    # --------------------------------------------------------
    # BENCHMARK ORB
    # --------------------------------------------------------

    (
        keypoints_orb,
        waktu_orb
    ) = benchmark_detector(
        orb,
        image,
        N_ULANGAN
    )

    # --------------------------------------------------------
    # JUMLAH KEYPOINT
    # --------------------------------------------------------

    jumlah_sift = len(
        keypoints_sift
    )

    jumlah_orb = len(
        keypoints_orb
    )

    # --------------------------------------------------------
    # TAMPILKAN HASIL
    # --------------------------------------------------------

    print(
        f"\nSIFT"
    )

    print(
        f"Jumlah keypoint : "
        f"{jumlah_sift}"
    )

    print(
        f"Waktu rata-rata : "
        f"{waktu_sift:.2f} ms"
    )

    print(
        f"\nORB"
    )

    print(
        f"Jumlah keypoint : "
        f"{jumlah_orb}"
    )

    print(
        f"Waktu rata-rata : "
        f"{waktu_orb:.2f} ms"
    )

    # --------------------------------------------------------
    # HITUNG PERBANDINGAN KECEPATAN
    # --------------------------------------------------------

    if waktu_orb > 0:

        perbandingan = (
            waktu_sift /
            waktu_orb
        )

        print(
            f"\nORB sekitar "
            f"{perbandingan:.2f}x "
            f"lebih cepat daripada SIFT."
        )

    else:

        perbandingan = 0

    # ========================================================
    # VISUALISASI KEYPOINT
    # ========================================================

    # SIFT
    visual_sift = cv2.drawKeypoints(
        image,
        keypoints_sift,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # ORB
    visual_orb = cv2.drawKeypoints(
        image,
        keypoints_orb,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # --------------------------------------------------------
    # FIGURE SIFT VS ORB
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].imshow(
        visual_sift,
        cmap="gray"
    )

    axes[0].set_title(
        f"SIFT\n"
        f"{jumlah_sift} keypoint\n"
        f"{waktu_sift:.2f} ms"
    )

    axes[0].axis("off")

    axes[1].imshow(
        visual_orb,
        cmap="gray"
    )

    axes[1].set_title(
        f"ORB\n"
        f"{jumlah_orb} keypoint\n"
        f"{waktu_orb:.2f} ms"
    )

    axes[1].axis("off")

    fig.suptitle(
        "Tugas 2 - Perbandingan SIFT dan ORB",
        fontsize=15
    )

    plt.tight_layout()

    output_keypoint = (
        output_path /
        "tugas2_keypoints_sift_vs_orb.png"
    )

    plt.savefig(
        output_keypoint,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n[OK] Visualisasi keypoint disimpan:"
        f"\n    {output_keypoint}"
    )

    # ========================================================
    # GRAFIK BENCHMARK
    # ========================================================

    nama_detector = [
        "SIFT",
        "ORB"
    ]

    jumlah_keypoint = [
        jumlah_sift,
        jumlah_orb
    ]

    waktu = [
        waktu_sift,
        waktu_orb
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    # --------------------------------------------------------
    # GRAFIK JUMLAH KEYPOINT
    # --------------------------------------------------------

    axes[0].bar(
        nama_detector,
        jumlah_keypoint
    )

    axes[0].set_title(
        "Jumlah Keypoint"
    )

    axes[0].set_ylabel(
        "Jumlah Keypoint"
    )

    for i, value in enumerate(
        jumlah_keypoint
    ):

        axes[0].text(
            i,
            value,
            str(value),
            ha="center",
            va="bottom"
        )

    # --------------------------------------------------------
    # GRAFIK WAKTU
    # --------------------------------------------------------

    axes[1].bar(
        nama_detector,
        waktu
    )

    axes[1].set_title(
        f"Waktu Eksekusi Rata-rata\n"
        f"{N_ULANGAN} Pengulangan"
    )

    axes[1].set_ylabel(
        "Waktu (ms)"
    )

    for i, value in enumerate(
        waktu
    ):

        axes[1].text(
            i,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_benchmark = (
        output_path /
        "tugas2_benchmark_sift_vs_orb.png"
    )

    plt.savefig(
        output_benchmark,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n[OK] Grafik benchmark disimpan:"
        f"\n    {output_benchmark}"
    )

    # --------------------------------------------------------
    # RETURN HASIL
    # --------------------------------------------------------

    return {
        "jumlah_sift": jumlah_sift,
        "jumlah_orb": jumlah_orb,
        "waktu_sift": waktu_sift,
        "waktu_orb": waktu_orb,
        "perbandingan": perbandingan
    }


# ============================================================
# TUGAS 3
# MEMBUAT LAPORAN
# ============================================================

def buat_laporan(
    output_path,
    hasil_tugas1,
    hasil_tugas2
):

    print("\n")
    print("=" * 60)
    print("TUGAS 3 - MEMBUAT LAPORAN")
    print("=" * 60)

    # --------------------------------------------------------
    # Mengambil hasil Canny
    # --------------------------------------------------------

    canny_text = ""

    for data in hasil_tugas1:

        (
            nama,
            threshold_low,
            threshold_high,
            hasil,
            jumlah_tepi,
            persentase
        ) = data

        canny_text += (
            f"| {nama} | "
            f"{threshold_low}-{threshold_high} | "
            f"{jumlah_tepi} | "
            f"{persentase:.2f}% |\n"
        )

    # --------------------------------------------------------
    # Isi laporan
    # --------------------------------------------------------

    laporan = f"""# Laporan Praktikum Minggu 3
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
{canny_text}

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

Benchmark dilakukan sebanyak **{N_ULANGAN} kali pengulangan** pada citra yang sama.

| Detektor | Jumlah Keypoint | Waktu Eksekusi Rata-rata |
|---|---:|---:|
| SIFT | {hasil_tugas2["jumlah_sift"]} | {hasil_tugas2["waktu_sift"]:.2f} ms |
| ORB | {hasil_tugas2["jumlah_orb"]} | {hasil_tugas2["waktu_orb"]:.2f} ms |

Berdasarkan hasil benchmark, waktu eksekusi SIFT adalah **{hasil_tugas2["waktu_sift"]:.2f} ms**, sedangkan ORB adalah **{hasil_tugas2["waktu_orb"]:.2f} ms**.

Pada pengujian ini, ORB sekitar **{hasil_tugas2["perbandingan"]:.2f} kali** lebih cepat daripada SIFT.

Jumlah keypoint SIFT adalah **{hasil_tugas2["jumlah_sift"]}**, sedangkan ORB adalah **{hasil_tugas2["jumlah_orb"]}**.

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
"""

    # --------------------------------------------------------
    # Simpan laporan
    # --------------------------------------------------------

    file_laporan = (
        output_path /
        "laporan_praktikum_minggu3.md"
    )

    file_laporan.write_text(
        laporan,
        encoding="utf-8"
    )

    print(
        f"\n[OK] Laporan dibuat:"
        f"\n    {file_laporan}"
    )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("PRAKTIKUM MINGGU 3")
    print("DETEKSI TEPI & EKSTRAKSI FITUR CITRA")
    print("=" * 60)

    # --------------------------------------------------------
    # Membuat folder output
    # --------------------------------------------------------

    output_path = buat_folder_output()

    # --------------------------------------------------------
    # Membaca gambar
    # --------------------------------------------------------

    image = baca_gambar(
        IMAGE_PATH
    )

    # --------------------------------------------------------
    # Simpan citra asli
    # --------------------------------------------------------

    simpan_gambar_asli(
        image,
        output_path
    )

    # --------------------------------------------------------
    # TUGAS 1
    # --------------------------------------------------------

    hasil_tugas1 = tugas1_deteksi_tepi(
        image,
        output_path
    )

    # --------------------------------------------------------
    # TUGAS 2
    # --------------------------------------------------------

    hasil_tugas2 = tugas2_sift_vs_orb(
        image,
        output_path
    )

    # --------------------------------------------------------
    # TUGAS 3
    # --------------------------------------------------------

    buat_laporan(
        output_path,
        hasil_tugas1,
        hasil_tugas2
    )

    # --------------------------------------------------------
    # SELESAI
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PRAKTIKUM SELESAI")
    print("=" * 60)

    print(
        f"\nSemua hasil berada di:"
        f"\n{output_path.resolve()}"
    )

    print("\nFile yang dihasilkan:")

    for file in output_path.iterdir():

        if file.is_file():

            print(
                f"  - {file.name}"
            )


# ============================================================
# EKSEKUSI
# ============================================================

if __name__ == "__main__":

    main()