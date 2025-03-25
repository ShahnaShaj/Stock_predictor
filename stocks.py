import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Caching function to fetch stock data
@st.cache_data
def fetch_data(ticker):
    df = yf.download(ticker, start='2015-01-01', end='2025-03-23', auto_adjust=True)
    return df

# Caching function to preprocess data
@st.cache_data
def preprocess_data(df):
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    train_size = math.ceil(len(data) * 0.9)
    train_data = scaled_data[:train_size]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, train_size, scaler, scaled_data

# Caching function to train the model and save it
@st.cache_resource
def get_trained_model(x_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.fit(x_train, y_train, batch_size=32, epochs=30, callbacks=[early_stop])
    model.save('stock_model.h5')  # Save the model after training
    return model

# Load model if it exists
def load_model_if_exists():
    try:
        model = load_model('stock_model.h5')
        return model
    except:
        return None

# Streamlit App Title
st.title('📈 Stock Trend Predictor')

# Sidebar Ticker Selection
st.sidebar.header("🗂️ Stock Selector")
tickers = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Intel (INTC)": "INTC",
    "AMD (AMD)": "AMD"
}
selected_name = st.sidebar.selectbox("Choose a stock", list(tickers.keys()))
ticker = tickers[selected_name]

# Optional custom ticker input
custom_ticker = st.sidebar.text_input("Or enter a custom ticker", "")
if custom_ticker:
    ticker = custom_ticker.upper()

# Fetch and preprocess data
df = fetch_data(ticker)
x_train, y_train, train_size, scaler, scaled_data = preprocess_data(df)

# Display Data Summary
st.subheader('📊 Data Summary (2015–2025)')
st.dataframe(df.describe(), use_container_width=True)

# Closing Price Chart
st.subheader('📉 Closing Price vs Time')
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['Close'], label='Closing Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Moving Averages Chart
st.subheader('📊 Moving Averages (100MA & 200MA)')
df['100MA'] = df['Close'].rolling(100).mean()
df['200MA'] = df['Close'].rolling(200).mean()
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['Close'], label='Close')
ax.plot(df['100MA'], label='100-Day MA')
ax.plot(df['200MA'], label='200-Day MA')
ax.legend()
st.pyplot(fig)

# Load or train the model
model = load_model_if_exists()

if model is None:  # If no model is cached, train a new one
    model = get_trained_model(x_train, y_train)

# Testing Data
x_test, y_test = [], df[['Close']].values[train_size:]
test_data = scaled_data[train_size - 60:]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Metrics
y_test_actual = df[['Close']].values[train_size:train_size + len(predictions)]
rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)

st.markdown(f"📏 **Root Mean Squared Error (RMSE):** `{rmse:.4f}`")
st.markdown(f"📐 **Mean Absolute Error (MAE):** `{mae:.4f}`")

# Align predictions
valid = df.iloc[train_size:].copy()
valid['Predictions'] = np.nan
valid.iloc[:len(predictions), valid.columns.get_loc('Predictions')] = predictions.flatten()

# Predictions vs Actual Plot
st.subheader('📈 Predictions vs Actual')
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index[:train_size], df['Close'][:train_size], label='Train')
ax.plot(df.index[train_size:], df['Close'][train_size:], label='Actual')
ax.plot(valid.index[:len(predictions)], valid['Predictions'].dropna(), label='Predictions', linestyle='dashed')
ax.legend()
st.pyplot(fig)

# Predictions Table
st.subheader('📋 Actual vs Predicted Prices')
valid_display = valid[['Close', 'Predictions']].dropna()
valid_display = valid_display.rename(columns={'Close': 'Actual Price', 'Predictions': 'Predicted Price'})
valid_display.index.name = 'Date'
st.dataframe(valid_display, use_container_width=True)

# Future Prediction (Next 60 Days)
st.subheader('🔮 Forecast: Next 60 Days')
future_data = df[['Close']].values[-60:].copy()
future_scaled = scaler.transform(future_data)
x_future = [future_scaled]
x_future = np.array(x_future)
x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

future_predictions = []
for _ in range(60):
    pred = model.predict(x_future)
    future_predictions.append(pred[0][0])
    x_future = np.append(x_future[:,1:,:], np.reshape(pred, (1,1,1)), axis=1)
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

# Plot Future
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(range(1, 61), future_predictions, label='Predicted Prices')
ax.set_xlabel('Days Ahead')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Display Future Predictions Table
future_df = pd.DataFrame(future_predictions, columns=['Predicted Price'])
future_df.index = pd.date_range(start=df.index[-1], periods=60, freq='D')
future_df.index.name = 'Date'
st.dataframe(future_df, use_container_width=True)