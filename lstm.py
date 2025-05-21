import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------- CONFIGURATION ----------------------
start = "1996-02-02"
end = "2024-12-30"
csv_path = "RELIANCE.NS_stock_data.csv"  # Ensure file is updated
window_size = 120

# ---------------------- DATA LOADING ----------------------
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
df = df[(df['Date'] >= start) & (df['Date'] <= end)]
df.reset_index(drop=True, inplace=True)

print("\nAvailable columns:", df.columns.tolist())
print("\nInitial dataset (first 5 rows):")
print(df.head())
print("\nInitial dataset (last 5 rows):")
print(df.tail())

# Check for duplicates
print("\nChecking for duplicate dates...")
duplicate_dates = df[df.duplicated(subset='Date')]
print(f"Found {len(duplicate_dates)} duplicate dates")
if len(duplicate_dates) > 0:
    print(duplicate_dates)

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# ---------------------- FEATURE ENGINEERING ----------------------
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['Return'] = df['Close'].pct_change()
df['Volatility'] = df['Return'].rolling(window=20).std()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df['RSI'] = RSI(df['Close'])

for lag in range(1, 6):
    df[f'Lag_{lag}'] = df['Close'].shift(lag)

df.dropna(inplace=True)

# ---------------------- SCALING ----------------------
features = ['Close', 'MA20', 'MA50', 'Volatility', 'RSI', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']
data = df[features].values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ---------------------- SEQUENCE CREATION ----------------------
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, window_size)

# ---------------------- SPLITTING ----------------------
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

# ---------------------- MODEL BUILDING ----------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# ---------------------- CALLBACKS ----------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = os.path.join(log_dir, f"final_LSTM_model_{timestamp}.keras")

early_stop = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# ---------------------- TRAINING ----------------------
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)

# ---------------------- LOAD BEST MODEL ----------------------
model = load_model(model_path)

# ---------------------- PREDICTION ----------------------
train_pred = model.predict(X_train).reshape(-1, 1)
val_pred = model.predict(X_val).reshape(-1, 1)
test_pred = model.predict(X_test).reshape(-1, 1)

def inverse_transform(preds):
    zeros = np.zeros((len(preds), scaled_data.shape[1] - 1))
    full = np.concatenate([preds, zeros], axis=1)
    return scaler.inverse_transform(full)[:, 0]

train_pred_actual = inverse_transform(train_pred)
val_pred_actual = inverse_transform(val_pred)
test_pred_actual = inverse_transform(test_pred)

train_y_actual = inverse_transform(y_train.reshape(-1, 1))
val_y_actual = inverse_transform(y_val.reshape(-1, 1))
test_y_actual = inverse_transform(y_test.reshape(-1, 1))

# ---------------------- EVALUATION ----------------------
def calculate_metrics(actual, pred):
    return {
        'MSE': mean_squared_error(actual, pred),
        'RMSE': np.sqrt(mean_squared_error(actual, pred)),
        'MAE': mean_absolute_error(actual, pred),
        'R2': r2_score(actual, pred),
        'MAPE': np.mean(np.abs((actual - pred) / actual)) * 100,
        'PMAE': np.mean(np.abs((actual - pred) / pred)) * 100
    }

metrics_train = calculate_metrics(train_y_actual, train_pred_actual)
metrics_val = calculate_metrics(val_y_actual, val_pred_actual)
metrics_test = calculate_metrics(test_y_actual, test_pred_actual)

# ---------------------- SAVE METRICS AND PLOTS ----------------------
metrics_file = os.path.join(log_dir, f"metrics_LSTM_{timestamp}.txt")
with open(metrics_file, "w") as f:
    f.write(f"Model Performance Metrics ({timestamp})\n")
    f.write("="*60 + "\n")
    for name, metric in zip(['Train', 'Validation', 'Test'], [metrics_train, metrics_val, metrics_test]):
        f.write(f"{name} Metrics:\n")
        for k, v in metric.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\n")

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(log_dir, f"loss_curve_LSTM_{timestamp}.png"))
plt.close()

# Metrics Comparison
metric_names = list(metrics_train.keys())
train_values = list(metrics_train.values())
val_values = list(metrics_val.values())
test_values = list(metrics_test.values())

x = np.arange(len(metric_names))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, train_values, width, label='Train')
plt.bar(x, val_values, width, label='Validation')
plt.bar(x + width, test_values, width, label='Test')
plt.ylabel('Score')
plt.title('Performance Metrics Comparison')
plt.xticks(x, metric_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(log_dir, f"metrics_comparison_LSTM_{timestamp}.png"))
plt.close()

# ---------------------- FULL PREDICTION ON ALL DATA ----------------------
# Predict on full dataset
full_pred_scaled = model.predict(X).reshape(-1, 1)

# Inverse transform predictions and actual Close prices
full_pred_actual = inverse_transform(full_pred_scaled)
full_actual_y = inverse_transform(y.reshape(-1, 1))

# Get matching date range
full_dates = df['Date'].iloc[window_size:].reset_index(drop=True)

# Plot Actual vs. Predicted
plt.figure(figsize=(15, 6))
plt.plot(full_dates, full_actual_y, label='Actual Close', color='black')
plt.plot(full_dates, full_pred_actual, label='Predicted Close', color='blue', linestyle='--')
plt.title('Full Range Actual vs. Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save plot
plot_path = os.path.join(log_dir, f"full_actual_vs_pred_{timestamp}.png")
plt.savefig(plot_path)
plt.close()
