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
4. **Deployment**: Streamlit + Heroku.

## Menjalankan Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Heroku
```bash
heroku login
heroku create adib-mlapp
git init
heroku git:remote -a adib-mlapp
git add .
git commit -m "deploy"
git push heroku master
```
