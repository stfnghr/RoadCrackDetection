# 🛣️ Road Crack Detection Using Digital Image Processing

> Sistem deteksi retakan jalan berbasis **Computer Vision** yang memproses foto dari perangkat mobile secara otomatis, menganalisis tingkat keparahan kerusakan, dan mengkategorikannya secara real-time.

---

## 📋 Deskripsi Proyek

Proyek ini adalah implementasi sistem **Road Crack Detection** menggunakan teknik Digital Image Processing dengan OpenCV dan FastAPI. Sistem dirancang untuk mendeteksi tingkat kerusakan jalan melalui foto yang diambil dari perangkat mobile (iOS), kemudian melakukan kategorisasi otomatis berdasarkan persentase luas kerusakan yang terdeteksi.

Dibuat untuk keperluan **Assessment of Learning Performance (ALP)** mata kuliah Digital Image Processing, Jurusan Informatics — **Universitas Ciputra Surabaya**.

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|---|---|
| **Backend** | Python, FastAPI, OpenCV, NumPy |
| **Frontend** | iOS — Swift, SwiftUI |
| **Communication** | REST API via IP Address (Local Network) |

---

## 📂 Struktur Proyek

```
RoadCrackDetection/
├── backend/
│   ├── main.py              # Main entry point & logika pemrosesan citra
│   ├── requirements.txt     # Daftar dependensi Python
│   └── ...
├── RoadCrackApp/
│   ├── ResultView.swift     # Tampilan hasil & logika status kategorisasi
│   └── NetworkManager.swift # Pengatur koneksi ke server backend
└── README.md
```

---

## 🚀 Panduan Instalasi & Menjalankan Server

### 1. Persiapan Backend (Mac)

Clone repositori dan masuk ke folder backend:

```bash
cd backend
```

Aktifkan virtual environment lalu instal seluruh dependensi:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Jalankan server API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server akan berjalan di `http://0.0.0.0:8000`.

---

### 2. Koneksi ke Aplikasi iOS

Sistem ini berjalan secara **local network (direct connection)** antara iPhone dan Mac.

1. Pastikan **iPhone dan Mac berada di jaringan Wi-Fi yang sama**.
2. Cek IP Address Mac kamu melalui **System Settings → Wi-Fi → Details** (contoh: `192.168.x.x`).
3. Buka proyek Xcode, lalu update URL endpoint di `NetworkManager.swift`:

```swift
private let serverURL = "http://[IP_ADDRESS_MAC_KAMU]:8000/analyze"
```

4. Build dan jalankan aplikasi ke perangkat iPhone.

---

## 📊 Alur Analisis Gambar

Sistem memproses setiap gambar melalui pipeline berikut:

```
Input Foto
    │
    ▼
① Preprocessing
   └─ Grayscale conversion, Median Blur, CLAHE
    │
    ▼
② Segmentation
   └─ Adaptive Thresholding, Morphological Operations (Opening & Closing)
    │
    ▼
③ Detection
   └─ Deteksi kontur dengan filter area & solidity
    │
    ▼
④ Severity Calculation
   └─ Perhitungan luas kerusakan menggunakan Convex Hull
    │
    ▼
⑤ Categorization & Output
```

---

## 🔍 Kategori Kerusakan

Hasil analisis dikategorikan secara otomatis berdasarkan persentase luas area retakan:

| Status | Kategori | Threshold |
|:---:|---|---|
| 🟢 | **Ringan** | Severity `< 15%` |
| 🟠 | **Sedang** | `15%` ≤ Severity `< 30%` |
| 🔴 | **Rusak Parah** | Severity `≥ 30%` |

---

## 📜 Lisensi

Proyek ini dibuat semata-mata untuk keperluan akademis — **Assessment of Learning Performance (ALP)** mata kuliah Digital Image Processing, Jurusan Informatics, Universitas Ciputra Surabaya. Tidak untuk dipublikasikan atau digunakan secara komersial.
