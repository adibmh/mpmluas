# Prediksi Nominal Transaksi UPI

## Deskripsi
Aplikasi machine learning untuk memprediksi nominal transaksi UPI berdasarkan waktu dan identitas pengguna.

## Dataset
Dataset transaksi individual UPI berisi kolom:
- Timestamp
- Sender Name & Sender UPI ID
- Receiver Name & Receiver UPI ID
- Amount (INR)
- Status

## Alur Proses
1. **EDA**: Visualisasi distribusi transaksi & status.
2. **Preprocessing**: Encoding, scaling, split data.
3. **Training Model**: Linear Regression, Decision Tree, SVR → SVR terbaik.
4. **Deployment**: Heroku.

