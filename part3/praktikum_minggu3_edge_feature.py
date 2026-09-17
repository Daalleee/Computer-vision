#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Praktikum Minggu 3 — Deteksi Tepi & Ekstraksi Fitur Citra
Mata Kuliah: Computer Vision

Skrip ini mengimplementasikan tiga poin tugas praktikum pada slide Minggu 3:

    1. Menerapkan Sobel & Canny pada sebuah citra, dan membandingkan hasil
       Canny pada 3 setelan threshold (T_rendah, T_tinggi) yang berbeda.
    2. Mengekstrak keypoint dengan SIFT dan ORB pada citra yang sama, lalu
       membandingkan jumlah keypoint yang terdeteksi dan waktu eksekusi
       rata-rata (dibenchmark selama N_ULANGAN kali agar hasil stabil).
    3. Menghasilkan berkas `laporan_template.md` berisi kerangka laporan
       (termasuk pertanyaan reflektif "kapan sebaiknya memilih ORB
       dibanding SIFT?") yang WAJIB diisi dan dijawab sendiri oleh
       mahasiswa — bagian analisis/kesimpulan sengaja dikosongkan agar
       penalaran akademis tetap berasal dari mahasiswa, bukan dari skrip.

Cara pakai
----------
    python praktikum_minggu3_edge_feature.py --image path/ke/citra.jpg
    python praktikum_minggu3_edge_feature.py            # tanpa --image,
                                                          # otomatis memakai
                                                          # citra uji standar
                                                          # (skimage) agar
                                                          # skrip tetap bisa
                                                          # dijalankan tanpa
                                                          # berkas eksternal

Dependensi: opencv-python, numpy, matplotlib, scikit-image (opsional,
hanya dipakai sebagai fallback citra contoh jika --image tidak diberikan).

Catatan perangkat edge (Raspberry Pi dkk.): bagian SIFT pada skrip ini
cukup berat secara komputasi untuk mikrokontroler/SBC kelas Raspberry Pi.
Jika target akhir adalah aplikasi real-time di perangkat edge, gunakan
hasil benchmark pada Tugas 2 sebagai bukti kuantitatif untuk memilih ORB.

Penulis kerangka kode: dibuatkan sebagai bahan ajar MK Computer Vision.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # aman untuk dijalankan headless (server/SSH/RPi tanpa GUI)
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CannySetting:
    """Satu pasang threshold (T_rendah, T_tinggi) untuk cv2.Canny beserta labelnya."""
    label: str
    t_rendah: int
    t_tinggi: int


# Tiga setelan threshold Canny yang diminta pada tugas praktikum.
# Silakan ubah nilai ini sesuai karakteristik citra Anda sendiri.
CANNY_SETTINGS: list[CannySetting] = [
    CannySetting("Longgar (deteksi banyak tepi, rentan noise)", 30, 90),
    CannySetting("Sedang (baseline umum)", 100, 200),
    CannySetting("Ketat (hanya tepi paling kuat)", 150, 250),
]

N_ULANGAN_BENCHMARK = 20  # jumlah pengulangan untuk merata-ratakan waktu eksekusi
JUMLAH_FITUR_MAKS = 500   # nfeatures untuk ORB, dan pembatas jumlah keypoint SIFT yang divisualisasikan


# ---------------------------------------------------------------------------
# Utilitas umum
# ---------------------------------------------------------------------------

def muat_citra(path: str | None) -> np.ndarray:
    """Memuat citra grayscale dari `path`.

    Jika `path` tidak diberikan (None), skrip memakai citra uji standar
    dari scikit-image ("cameraman") sehingga tetap bisa dijalankan tanpa
    berkas eksternal apa pun — berguna untuk demo cepat atau CI.
    """
    if path:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Tidak dapat membaca citra pada path: {path}")
        return img

    try:
        from skimage import data  # import lokal: hanya dibutuhkan untuk fallback
    except ImportError as exc:
        raise RuntimeError(
            "Tidak ada --image yang diberikan dan scikit-image tidak terpasang "
            "untuk memuat citra contoh. Instal dengan `pip install scikit-image` "
            "atau berikan --image path/ke/citra.jpg"
        ) from exc

    print("[INFO] --image tidak diberikan, memakai citra contoh skimage.data.camera()")
    return data.camera()


def siapkan_direktori_output(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Tugas 1 — Sobel & Canny pada 3 setelan threshold
# ---------------------------------------------------------------------------

def hitung_sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Menghitung magnitudo gradien Sobel |grad I| = sqrt(Gx^2 + Gy^2), dinormalisasi ke 0-255."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def tugas1_deteksi_tepi(img: np.ndarray, out_dir: Path) -> None:
    """Tugas 1: bandingkan Sobel (baseline) dengan Canny pada 3 setelan threshold."""
    print("\n=== TUGAS 1: Deteksi Tepi (Sobel vs Canny, 3 threshold) ===")

    sobel_mag = hitung_sobel_magnitude(img)
    hasil_canny = [
        (setting, cv2.Canny(img, setting.t_rendah, setting.t_tinggi))
        for setting in CANNY_SETTINGS
    ]

    # --- figur perbandingan: citra asli, Sobel, dan 3 varian Canny ---
    n_panel = 2 + len(hasil_canny)
    fig, axes = plt.subplots(1, n_panel, figsize=(4 * n_panel, 5.2))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Citra asli")

    axes[1].imshow(sobel_mag, cmap="gray")
    axes[1].set_title("Sobel |∇I|")

    for ax, (setting, tepi) in zip(axes[2:], hasil_canny):
        ax.imshow(tepi, cmap="gray")
        ax.set_title(f"Canny\nT=({setting.t_rendah},{setting.t_tinggi})\n{setting.label}", fontsize=9)
        jumlah_piksel_tepi = int(np.count_nonzero(tepi))
        persen = 100 * jumlah_piksel_tepi / tepi.size
        ax.set_xlabel(f"{jumlah_piksel_tepi} px tepi ({persen:.2f}%)", fontsize=8)
        print(
            f"  Canny {setting.label:45s} T=({setting.t_rendah:3d},{setting.t_tinggi:3d}) "
            f"-> {jumlah_piksel_tepi:6d} piksel tepi ({persen:5.2f}% dari citra)"
        )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Tugas 1 — Perbandingan Sobel vs Canny pada 3 Setelan Threshold", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path = out_dir / "tugas1_sobel_vs_canny.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Figur disimpan ke: {out_path}")


# ---------------------------------------------------------------------------
# Tugas 2 — SIFT vs ORB: jumlah keypoint & waktu eksekusi
# ---------------------------------------------------------------------------

def benchmark_detektor(detector, img: np.ndarray, n_ulangan: int) -> tuple[list, float]:
    """Menjalankan detectAndCompute berulang kali dan mengembalikan (keypoints, rata-rata_ms)."""
    # Jalankan sekali di luar hitungan waktu untuk "pemanasan" (warm-up)
    keypoints, _ = detector.detectAndCompute(img, None)

    t0 = time.perf_counter()
    for _ in range(n_ulangan):
        detector.detectAndCompute(img, None)
    total_detik = time.perf_counter() - t0

    rata_rata_ms = (total_detik / n_ulangan) * 1000.0
    return keypoints, rata_rata_ms


def tugas2_perbandingan_fitur(img: np.ndarray, out_dir: Path) -> None:
    """Tugas 2: bandingkan SIFT vs ORB dari sisi jumlah keypoint & waktu eksekusi."""
    print("\n=== TUGAS 2: Perbandingan SIFT vs ORB ===")

    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures=JUMLAH_FITUR_MAKS)

    kp_sift, ms_sift = benchmark_detektor(sift, img, N_ULANGAN_BENCHMARK)
    kp_orb, ms_orb = benchmark_detektor(orb, img, N_ULANGAN_BENCHMARK)

    print(f"  SIFT: {len(kp_sift):4d} keypoint  |  {ms_sift:7.2f} ms/citra  (rata-rata {N_ULANGAN_BENCHMARK}x)")
    print(f"  ORB : {len(kp_orb):4d} keypoint  |  {ms_orb:7.2f} ms/citra  (rata-rata {N_ULANGAN_BENCHMARK}x)")
    if ms_orb > 0:
        print(f"  -> ORB lebih cepat {ms_sift / ms_orb:.1f}x dibanding SIFT pada mesin ini")

    # --- visualisasi keypoint berdampingan ---
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis_sift = cv2.drawKeypoints(
        img_bgr, kp_sift, None, color=(0, 210, 216),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    vis_orb = img_bgr.copy()
    for kp in kp_orb:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(vis_orb, (x, y), 4, (0, 90, 216), thickness=-1, lineType=cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(cv2.cvtColor(vis_sift, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"SIFT — {len(kp_sift)} keypoint\n{ms_sift:.2f} ms/citra")
    axes[1].imshow(cv2.cvtColor(vis_orb, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"ORB — {len(kp_orb)} keypoint\n{ms_orb:.2f} ms/citra")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle("Tugas 2 — Keypoint SIFT vs ORB pada Citra yang Sama", fontsize=13, y=0.99)
    out_keypoints = out_dir / "tugas2_keypoints_sift_vs_orb.png"
    fig.savefig(out_keypoints, dpi=150)
    plt.close(fig)
    print(f"[OK] Figur keypoint disimpan ke: {out_keypoints}")

    # --- grafik batang: jumlah keypoint & waktu eksekusi ---
    fig, (ax_kp, ax_waktu) = plt.subplots(1, 2, figsize=(9, 4))

    ax_kp.bar(["SIFT", "ORB"], [len(kp_sift), len(kp_orb)], color=["#00B4D8", "#FF6B35"])
    ax_kp.set_title("Jumlah Keypoint Terdeteksi")
    ax_kp.set_ylabel("Jumlah keypoint")
    for i, v in enumerate([len(kp_sift), len(kp_orb)]):
        ax_kp.text(i, v, str(v), ha="center", va="bottom")

    ax_waktu.bar(["SIFT", "ORB"], [ms_sift, ms_orb], color=["#00B4D8", "#FF6B35"])
    ax_waktu.set_title(f"Waktu Eksekusi Rata-rata\n({N_ULANGAN_BENCHMARK}x pengulangan)")
    ax_waktu.set_ylabel("Waktu (ms/citra)")
    for i, v in enumerate([ms_sift, ms_orb]):
        ax_waktu.text(i, v, f"{v:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    out_bench = out_dir / "tugas2_benchmark_sift_vs_orb.png"
    fig.savefig(out_bench, dpi=150)
    plt.close(fig)
    print(f"[OK] Grafik benchmark disimpan ke: {out_bench}")

    return {
        "jumlah_sift": len(kp_sift),
        "jumlah_orb": len(kp_orb),
        "ms_sift": ms_sift,
        "ms_orb": ms_orb,
    }


# ---------------------------------------------------------------------------
# Tugas 3 — Kerangka laporan (WAJIB diisi mahasiswa, bukan oleh skrip)
# ---------------------------------------------------------------------------

def tulis_kerangka_laporan(out_dir: Path, hasil_tugas2: dict) -> None:
    """Menulis `laporan_template.md`.

    PENTING: bagian analisis & kesimpulan SENGAJA dikosongkan (ditandai
    dengan `<ISI DI SINI>`). Angka hasil benchmark sudah diisi otomatis
    dari komputasi nyata, tetapi *interpretasi* dan *argumen* harus
    ditulis sendiri oleh mahasiswa sebagai bagian dari penilaian.
    """
    isi = f"""# Laporan Praktikum Minggu 3 — Deteksi Tepi & Ekstraksi Fitur

**Nama       :** <ISI DI SINI>
**NIM        :** <ISI DI SINI>
**Mata Kuliah:** Computer Vision

## 1. Deteksi Tepi (Sobel vs Canny)

Lampirkan `tugas1_sobel_vs_canny.png`. Jelaskan perbedaan hasil pada
ketiga setelan threshold Canny yang diuji:

- <ISI DI SINI: pengaruh threshold longgar vs ketat terhadap jumlah tepi
  dan derau yang muncul>
- <ISI DI SINI: threshold mana yang paling sesuai untuk citra Anda, dan
  mengapa>

## 2. Perbandingan SIFT vs ORB

Hasil pengukuran otomatis pada citra uji (rata-rata {N_ULANGAN_BENCHMARK}x pengulangan):

| Detektor | Jumlah Keypoint | Waktu Eksekusi |
|----------|-----------------|----------------|
| SIFT     | {hasil_tugas2['jumlah_sift']} | {hasil_tugas2['ms_sift']:.2f} ms |
| ORB      | {hasil_tugas2['jumlah_orb']} | {hasil_tugas2['ms_orb']:.2f} ms |

Lampirkan `tugas2_keypoints_sift_vs_orb.png` dan
`tugas2_benchmark_sift_vs_orb.png`.

**Pertanyaan reflektif: kapan sebaiknya memilih ORB dibanding SIFT?**

<ISI DI SINI — jawab berdasarkan data di atas dan konteks aplikasi nyata,
misalnya: kebutuhan real-time, keterbatasan daya komputasi (mis.
Raspberry Pi/perangkat edge), kebutuhan akurasi tinggi, atau
pertimbangan lisensi.>

## 3. Kesimpulan

<ISI DI SINI>
"""
    out_path = out_dir / "laporan_template.md"
    out_path.write_text(isi, encoding="utf-8")
    print(f"\n[OK] Kerangka laporan disimpan ke: {out_path}")
    print("     -> Bagian '<ISI DI SINI>' WAJIB dilengkapi sendiri sebelum dikumpulkan.")


# ---------------------------------------------------------------------------
# Program utama
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Praktikum Minggu 3: Deteksi Tepi (Sobel/Canny) & Ekstraksi Fitur (SIFT/ORB)."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path ke citra input (grayscale/berwarna). Jika kosong, memakai citra contoh skimage.",
    )
    parser.add_argument(
        "--outdir", type=str, default="output",
        help="Direktori untuk menyimpan seluruh hasil (figur & laporan). Default: ./output",
    )
    args = parser.parse_args()

    img = muat_citra(args.image)
    out_dir = siapkan_direktori_output(args.outdir)

    tugas1_deteksi_tepi(img, out_dir)
    hasil_tugas2 = tugas2_perbandingan_fitur(img, out_dir)
    tulis_kerangka_laporan(out_dir, hasil_tugas2)

    print(f"\nSelesai. Semua berkas hasil ada di folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
