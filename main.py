"""
Prediksi Penjualan Produk UMKM
Metode : Regresi Linear & Random Forest Regression (supervised learning)
Output : metrik evaluasi, tabel & grafik prediksi vs aktual, prediksi bulan berikutnya
"""
import os

import matplotlib
matplotlib.use("Agg")  # supaya jalan juga tanpa layar (server/Colab)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/penjualan_umkm.csv"
OUT_DIR = "output"
TARGET = "jumlah_penjualan"
JUMLAH_LAG = 3          # pakai penjualan 1-3 bulan sebelumnya sebagai fitur
RASIO_UJI = 0.2         # 20% data paling akhir untuk testing
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- 1. Load data
def muat_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan dulu: python generate_data.py"
        )
    df = pd.read_csv(path, parse_dates=["bulan"])
    return df.sort_values("bulan").reset_index(drop=True)


# ------------------------------------------------------------ 2. Preprocessing
def preprocessing(df):
    print("Jumlah data      :", len(df))
    print("Missing value    :\n", df.isna().sum().to_string(), "\n")

    df = df.drop_duplicates(subset="bulan")
    df["harga"] = df["harga"].interpolate().bfill().ffill()  # isi missing value
    df[TARGET] = df[TARGET].interpolate()
    return df


def buat_fitur(df):
    """Fitur: kalender, harga, promo, dan penjualan bulan-bulan sebelumnya (lag)."""
    d = df.copy()
    d["bulan_ke"] = d["bulan"].dt.month
    d["tahun"] = d["bulan"].dt.year
    d["t"] = np.arange(len(d))  # indeks waktu -> menangkap tren naik
    for k in range(1, JUMLAH_LAG + 1):
        d[f"lag_{k}"] = d[TARGET].shift(k)
    d["rata2_3bln"] = d[TARGET].shift(1).rolling(3).mean()
    return d.dropna().reset_index(drop=True)


# ------------------------------------------------------------------ 3. Evaluasi
def evaluasi(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2}


# ------------------------------------------------------- 4. Prediksi bulan depan
def prediksi_bulan_depan(model, df_asli, fitur, harga_asumsi, promo_asumsi):
    """Bangun satu baris fitur untuk bulan berikutnya dari data terakhir."""
    terakhir = df_asli[TARGET].tolist()
    bulan_depan = df_asli["bulan"].iloc[-1] + pd.offsets.MonthBegin(1)
    baris = {
        "harga": harga_asumsi,
        "promo": promo_asumsi,
        "bulan_ke": bulan_depan.month,
        "tahun": bulan_depan.year,
        "t": len(df_asli),
        "rata2_3bln": np.mean(terakhir[-3:]),
    }
    for k in range(1, JUMLAH_LAG + 1):
        baris[f"lag_{k}"] = terakhir[-k]
    X_baru = pd.DataFrame([baris])[fitur]
    return bulan_depan, float(model.predict(X_baru)[0])


# ------------------------------------------------------------------------ main
def main():
    df = preprocessing(muat_data(DATA_PATH))
    data = buat_fitur(df)

    fitur = [c for c in data.columns if c not in ("bulan", TARGET)]
    X, y = data[fitur], data[TARGET]

    # Split berdasarkan waktu (BUKAN acak) agar tidak "mengintip" masa depan
    n_uji = int(len(data) * RASIO_UJI)
    X_latih, X_uji = X.iloc[:-n_uji], X.iloc[-n_uji:]
    y_latih, y_uji = y.iloc[:-n_uji], y.iloc[-n_uji:]
    tgl_uji = data["bulan"].iloc[-n_uji:]
    print(f"Data latih: {len(X_latih)} bulan | Data uji: {len(X_uji)} bulan\n")

    model = {
        "Regresi Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, random_state=42
        ),
    }

    hasil, prediksi = {}, {}
    for nama, m in model.items():
        m.fit(X_latih, y_latih)
        prediksi[nama] = m.predict(X_uji)
        hasil[nama] = evaluasi(y_uji.values, prediksi[nama])

    tabel_metrik = pd.DataFrame(hasil).T.round(3)
    print("=== Evaluasi Model (data uji) ===")
    print(tabel_metrik.to_string(), "\n")
    tabel_metrik.to_csv(f"{OUT_DIR}/metrik_evaluasi.csv")

    # Perbandingan prediksi vs aktual
    banding = pd.DataFrame({"bulan": tgl_uji.dt.strftime("%Y-%m"), "aktual": y_uji.values})
    for nama, p in prediksi.items():
        banding[f"prediksi_{nama}"] = p.round(1)
    print("=== Prediksi vs Aktual ===")
    print(banding.to_string(index=False), "\n")
    banding.to_csv(f"{OUT_DIR}/prediksi_vs_aktual.csv", index=False)

    # Grafik 1: aktual vs prediksi
    plt.figure(figsize=(10, 5))
    plt.plot(data["bulan"], y, label="Aktual", color="black", alpha=0.6)
    for nama, p in prediksi.items():
        plt.plot(tgl_uji, p, marker="o", label=f"Prediksi {nama}")
    plt.axvline(tgl_uji.iloc[0], color="gray", linestyle="--", label="Batas data latih/uji")
    plt.title("Prediksi vs Aktual Penjualan Produk UMKM")
    plt.xlabel("Bulan"); plt.ylabel("Jumlah penjualan (unit)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/grafik_prediksi_vs_aktual.png", dpi=150)
    plt.close()

    # Grafik 2: pentingnya fitur (Random Forest)
    rf = model["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=fitur).sort_values()
    plt.figure(figsize=(7, 4))
    imp.plot(kind="barh", color="steelblue")
    plt.title("Pentingnya Fitur (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/grafik_fitur_penting.png", dpi=150)
    plt.close()

    # Prediksi periode berikutnya pakai model terbaik (RMSE terkecil),
    # dilatih ulang dengan seluruh data
    terbaik = tabel_metrik["RMSE"].idxmin()
    final = model[terbaik].fit(X, y)
    harga_asumsi = df["harga"].iloc[-1]
    bulan_depan, nilai = prediksi_bulan_depan(
        final, df, fitur, harga_asumsi=harga_asumsi, promo_asumsi=0
    )
    print(f"Model terbaik: {terbaik}")
    print(f"Prediksi penjualan {bulan_depan:%Y-%m}: {nilai:.0f} unit "
          f"(asumsi: harga sama dengan bulan terakhir, tanpa promo)")

    pd.DataFrame(
        {"bulan": [bulan_depan.strftime("%Y-%m")], "model": [terbaik],
         "prediksi_penjualan": [round(nilai)]}
    ).to_csv(f"{OUT_DIR}/prediksi_bulan_depan.csv", index=False)


if __name__ == "__main__":
    main()
