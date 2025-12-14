# Program Klasifikasi Tekstur Permukaan

Program ini menggunakan Python untuk mengklasifikasikan tekstur permukaan bahan seperti plastik, kayu, dan besi berdasarkan gambar.

## Fitur

- Ekstraksi fitur tekstur menggunakan GLCM (Gray Level Co-occurrence Matrix)
- Klasifikasi menggunakan Support Vector Machine (SVM)
- Mendukung klasifikasi bahan seperti plastik, kayu, besi, dll.

## Persyaratan Sistem

- Python 3.7+
- Webcam atau gambar untuk testing

## Instalasi

1. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

2. Siapkan dataset:
   - Buat subfolder di folder `dataset/` untuk setiap kelas bahan (misalnya `plastik/`, `kayu/`, `besi/`)
   - Masukkan gambar JPG/PNG untuk setiap kelas

## Penggunaan

### Melatih Model

```
python main.py --train
```

### Mengklasifikasikan Gambar

```
python main.py --classify path/to/gambar.jpg
```

### Melihat Bantuan

```
python main.py --help
```

## Struktur Proyek

```
UAS PENGENALAN POLA 2.0/
├── main.py                 # Script utama
├── requirements.txt        # Dependensi Python
├── dataset/                # Folder dataset
│   ├── plastik/           # Gambar plastik
│   ├── kayu/              # Gambar kayu
│   └── besi/               # Gambar besi
└── README.md              # Dokumentasi ini
```

## Contoh Output

```
Memuat dataset...
Ditemukan 3 kelas: ['besi', 'kayu', 'plastik']
Memuat 50 gambar dari kelas 'besi'
Memuat 45 gambar dari kelas 'kayu'
Memuat 52 gambar dari kelas 'plastik'
Dataset dimuat: 147 sampel, 3 kelas
Akurasi model: 0.92
Model disimpan ke 'texture_classifier.pkl'

Gambar 'test.jpg' diklasifikasikan sebagai: plastik (probabilitas: 0.87)
```

## Catatan

- Pastikan dataset memiliki gambar yang cukup untuk setiap kelas (minimal 10 gambar per kelas)
- Gambar harus jelas dan fokus pada tekstur permukaan bahan
- Model akan disimpan sebagai `texture_classifier.pkl` setelah training
