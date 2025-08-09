from flask import Flask, render_template, request
import joblib
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)

# Load model, feature columns, dan scaler
model = joblib.load("best_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
scaler = joblib.load("scaler_amount.pkl")  # scaler untuk Amount (INR)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Ambil input dari form
    hour = int(request.form.get("hour", 0))
    weekday = int(request.form.get("weekday", 0))
    month = int(request.form.get("month", 1))
    status_binary = int(request.form.get("status_binary", 1))

    # Siapkan array input dengan ukuran sesuai feature_columns
    input_data = np.zeros(len(feature_columns))

    # Isi nilai numerik dasar
    if "Hour_" + str(hour) in feature_columns:
        idx = feature_columns.index("Hour_" + str(hour))
        input_data[idx] = 1
    if "Weekday_" + str(weekday) in feature_columns:
        idx = feature_columns.index("Weekday_" + str(weekday))
        input_data[idx] = 1
    if "Month_" + str(month) in feature_columns:
        idx = feature_columns.index("Month_" + str(month))
        input_data[idx] = 1

    # Status binary jika ada di fitur
    if "Status_binary" in feature_columns:
        idx = feature_columns.index("Status_binary")
        input_data[idx] = status_binary

    # Prediksi (scaled)
    pred_scaled = model.predict([input_data])[0]

    # Konversi kembali ke INR asli
    pred_original = scaler.inverse_transform([[pred_scaled]])[0][0]

    # Pastikan tidak negatif
    pred_original = max(0, pred_original)

    return render_template(
        "result.html",
        prediction_text=f"Prediksi Nominal Transaksi: ₹ {pred_original:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
