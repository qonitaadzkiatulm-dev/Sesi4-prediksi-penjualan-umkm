| **Nama** | Qonita Adzkiatul Mardiyah |
|---|---|
| **NPM** | G1A023086 |



# Prediksi Penjualan Produk UMKM

Studi kasus: memperkirakan penjualan bulan berikutnya berdasarkan data penjualan sebelumnya.
Metode: **Regresi Linear** dan **Random Forest Regression** (supervised learning).

## Alur
1. **Load data** - `data/penjualan_umkm.csv` (kolom: `bulan`, `harga`, `promo`, `jumlah_penjualan`)
2. **Preprocessing** - hapus duplikat, isi missing value (interpolasi)
3. **Feature engineering** - bulan, tahun, indeks waktu, penjualan 1-3 bulan sebelumnya (lag), rata-rata 3 bulan
4. **Split data berdasarkan waktu** - 80% awal untuk latih, 20% akhir untuk uji (bukan acak)
5. **Training & evaluasi** - MAE, RMSE, MAPE, R2
6. **Perbandingan prediksi vs aktual** + prediksi bulan berikutnya

## Cara menjalankan
```bash
pip install -r requirements.txt
python generate_data.py   # opsional: buat data simulasi (lewati jika punya data asli)
python main.py
```

## Output (folder `output/`)
- `metrik_evaluasi.csv` - perbandingan performa kedua model
- `prediksi_vs_aktual.csv` - tabel nilai prediksi dan aktual
- `grafik_prediksi_vs_aktual.png` - grafik perbandingan
- `grafik_fitur_penting.png` - fitur paling berpengaruh (Random Forest)
- `prediksi_bulan_depan.csv` - prediksi periode berikutnya

## Catatan
- Data bawaan adalah **data simulasi**. Ganti dengan data penjualan UMKM asli dengan format kolom yang sama.
- Random Forest tidak bisa memprediksi nilai di luar rentang data latih, sehingga kurang cocok untuk data dengan tren naik; Regresi Linear lebih unggul pada data seperti ini.
