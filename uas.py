# ==========================================
# Stage 1: Load & EDA
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("transactions.csv", sep=";")

# Konversi Timestamp ke datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Ekstraksi fitur waktu
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Weekday'] = df['Timestamp'].dt.weekday  # 0=Monday
df['Month'] = df['Timestamp'].dt.month

# Cek missing
print(df.isnull().sum())

# Distribusi target (Amount)
plt.figure()
plt.hist(df['Amount (INR)'].dropna(), bins=50)
plt.title("Distribusi Amount (INR)")
plt.xlabel("Amount")
plt.ylabel("Frekuensi")
plt.tight_layout()
plt.show()

# Proporsi status
plt.figure()
df['Status'].value_counts().plot(kind='bar')
plt.title("Status Transaksi")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ==========================================
# Stage 2: Preprocessing
# ==========================================
# Drop baris yang Timestamp-nya gagal parse
df = df.dropna(subset=['Timestamp'])

# Untuk regresi target = Amount
df = df[df['Amount (INR)'].notna()]

# Encode Status
df['Status_binary'] = df['Status'].apply(lambda x: 1 if str(x).strip().lower() in ['success','succeeded','completed'] else 0)

# Mengambil top 10 sender/receiver berdasarkan frekuensi
top_senders = df['Sender Name'].value_counts().nlargest(10).index
top_receivers = df['Receiver Name'].value_counts().nlargest(10).index

df['Sender_top'] = df['Sender Name'].where(df['Sender Name'].isin(top_senders), other='Other')
df['Receiver_top'] = df['Receiver Name'].where(df['Receiver Name'].isin(top_receivers), other='Other')

# One-hot encoding
categorical_cols = ['Sender_top', 'Receiver_top', 'Weekday', 'Month', 'Hour']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

from sklearn.preprocessing import StandardScaler

# Scaling Amount dan simpan scaler
scaler = StandardScaler()
df[['Amount (INR)']] = scaler.fit_transform(df[['Amount (INR)']])

# Simpan scaler
import joblib
joblib.dump(scaler, "scaler_amount.pkl")

from sklearn.model_selection import train_test_split

# Regresi
X_reg = df.drop(columns=['Transaction ID', 'Timestamp', 'Sender Name', 'Sender UPI ID',
                        'Receiver Name', 'Receiver UPI ID', 'Amount (INR)', 'Status', 'Status_binary'], errors='ignore')
y_reg = df['Amount (INR)']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Klasifikasi (opsional)
X_clf = X_reg
y_clf = df['Status_binary']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# ==========================================
# Stage 3: Training & Evaluasi
# ==========================================
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def eval_regression(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Train beberapa model
lr = LinearRegression().fit(X_train_reg, y_train_reg)
res_lr = eval_regression(lr, X_test_reg, y_test_reg)

dt = DecisionTreeRegressor(random_state=42).fit(X_train_reg, y_train_reg)
res_dt = eval_regression(dt, X_test_reg, y_test_reg)

svr_pipe = SVR().fit(X_train_reg, y_train_reg)
res_svr = eval_regression(svr_pipe, X_test_reg, y_test_reg)

# Bandingkan MAE dan R²
models = ['Linear', 'DecisionTree', 'SVR']
mae_vals = [res_lr['MAE'], res_dt['MAE'], res_svr['MAE']]
r2_vals = [res_lr['R2'], res_dt['R2'], res_svr['R2']]

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(models, mae_vals)
plt.title("MAE Comparison")
plt.subplot(1,2,2)
plt.bar(models, r2_vals)
plt.title("R² Comparison")
plt.tight_layout()
plt.show()

# Pilih model terbaik
best_score = -np.inf
best_model = None
model_dict = {
    'Linear': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'SVR': SVR()
}

for name, model in model_dict.items():
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    r2 = r2_score(y_test_reg, pred)
    if r2 > best_score:
        best_score = r2
        best_model = model

# Simpan model terbaik & kolom fitur
joblib.dump(best_model, "best_model.pkl")
joblib.dump(X_train_reg.columns.tolist(), "feature_columns.pkl")
print("✅ Model terbaik & scaler disimpan — Siap digunakan di Flask")
