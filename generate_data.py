"""Membuat data penjualan UMKM sintetis (simulasi) -> data/penjualan_umkm.csv

Data ini hanya simulasi untuk keperluan tugas. Jika punya data asli,
cukup ganti file CSV dengan kolom yang sama:
    bulan (YYYY-MM-DD), harga, promo (0/1), jumlah_penjualan
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

bulan = pd.date_range("2022-01-01", periods=60, freq="MS")  # 5 tahun, per bulan
n = len(bulan)

tren = np.linspace(100, 230, n)                       # penjualan naik pelan-pelan
musiman = 25 * np.sin(2 * np.pi * (bulan.month - 3) / 12)  # pola musiman tahunan
lonjakan = np.where(bulan.month.isin([4, 12]), 30, 0)      # momen Lebaran/Natal-Tahun Baru
promo = rng.binomial(1, 0.25, n)                      # 25% bulan ada promo
harga = 15000 + rng.normal(0, 500, n).round(-2)       # harga relatif stabil
noise = rng.normal(0, 8, n)

jumlah = tren + musiman + lonjakan + 20 * promo - 0.002 * (harga - 15000) + noise
df = pd.DataFrame({
    "bulan": bulan,
    "harga": harga.astype(int),
    "promo": promo,
    "jumlah_penjualan": jumlah.round().astype(int),
})

# sisipkan sedikit missing value agar tahap preprocessing bermakna
df.loc[[7, 21, 40], "harga"] = np.nan

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/penjualan_umkm.csv", index=False)
print(f"Data dibuat: {len(df)} baris -> data/penjualan_umkm.csv")
