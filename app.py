import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# LSTM imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Trying TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    print("✓ Using TensorFlow 2.x with integrated Keras")
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping
        print("✓ Using standalone Keras")
    except ImportError:
        print("❌ ERROR: Neither TensorFlow nor Keras found!")
        Sequential = None

# INITIALIZE APP
# static_url_path='' and static_folder='.' ensure files like 'silver face1.jpeg' are visible
app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# Global cache for LSTM model
lstm_model_cache = {
    'model': None,
    'scaler': None,
    'last_trained': None,
    'last_data': None
}

# --- STATIC FILE ROUTES (FOR IMAGES AND HTML) ---

@app.route('/')
def index():
    # Serves website.html from your main folder
    return send_from_directory(os.getcwd(), 'website.html')

@app.route('/<path:path>')
def serve_static(path):
    # Serves images (silver face1.jpeg), CSS, and JS files
    return send_from_directory(os.getcwd(), path)

# --- HELPER FUNCTIONS ---

def fetch_inr_data(ticker, start, end):
    try:
        metal = yf.download(ticker, start=start, end=end, progress=False)
        usd_inr = yf.download("USDINR=X", start=start, end=end, progress=False)
        df = pd.concat([metal["Close"], usd_inr["Close"]], axis=1, join="inner")
        df.columns = ["Metal_USD", "USD_INR"]
        df["Close_INR"] = df["Metal_USD"] * df["USD_INR"]
        df["Return"] = df["Close_INR"].pct_change() * 100

        def label_sentiment(change):
            if pd.isna(change): return "Birth"
            if change < -2: return "Death"
            elif change < -0.5: return "Reversal"
            elif change < 0.5: return "Birth"
            elif change < 2: return "Growth"
            else: return "Peak"

        df["Sentiment"] = df["Return"].apply(label_sentiment)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#222', edgecolor='none', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length):
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(100, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm_model(data, seq_length=60, epochs=50):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        X, y = create_sequences(scaled_data, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = build_lstm_model(seq_length)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, callbacks=[early_stop], verbose=0)
        return model, scaler
    except Exception as e:
        print(f"Error training LSTM: {e}")
        return None, None

def forecast_lstm(model, scaler, last_sequence, days=30):
    try:
        predictions = []
        current_sequence = last_sequence.copy()
        for _ in range(days):
            current_sequence_reshaped = current_sequence.reshape(1, len(current_sequence), 1)
            next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
            predictions.append(next_pred)
            current_sequence = np.append(current_sequence[1:], next_pred)
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"Error forecasting: {e}")
        return None

# --- API ENDPOINTS ---

@app.route('/api/chart/<period>')
def get_chart(period):
    end_date = datetime.now().strftime('%Y-%m-%d')
    period_map = {'weekly': 7, 'monthly': 30, 'quarterly': 90, 'half-yearly': 180}
    days = period_map.get(period.lower(), 30)
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    df = fetch_inr_data("SI=F", start_date, end_date)
    if df is None or len(df) == 0:
        return jsonify({"error": "No data"}), 500
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#222')
    ax.set_facecolor('#222')
    ax.plot(df.index, df['Close_INR'], color='#8b1e2d', linewidth=2.5)
    ax.tick_params(colors='white')
    img_str = plot_to_base64(fig)
    return jsonify({
        "image": img_str,
        "current_price": round(float(df['Close_INR'].iloc[-1]), 2),
        "price_change": round(float(df['Return'].iloc[-1]), 2),
        "sentiment": df['Sentiment'].iloc[-1]
    })

@app.route('/api/compare-gold')
def compare_gold():
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    silver_df = fetch_inr_data("SI=F", start_date, end_date)
    gold_df = fetch_inr_data("GC=F", start_date, end_date)
    if silver_df is None or gold_df is None:
        return jsonify({"error": "Could not fetch data"}), 500
    silver_norm = (silver_df['Close_INR'] / silver_df['Close_INR'].iloc[0]) * 100
    gold_norm = (gold_df['Close_INR'] / gold_df['Close_INR'].iloc[0]) * 100
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#222')
    ax.set_facecolor('#222')
    ax.plot(silver_norm.index, silver_norm, color='#c0c0c0', label='Silver')
    ax.plot(gold_norm.index, gold_norm, color='#ffd700', label='Gold')
    ax.tick_params(colors='white')
    ax.legend()
    img_str = plot_to_base64(fig)
    return jsonify({"image": img_str, "better_performer": "Silver" if silver_norm.iloc[-1] > gold_norm.iloc[-1] else "Gold"})

@app.route('/api/forecast')
def forecast():
    global lstm_model_cache
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    df = fetch_inr_data("SI=F", start_date, end_date)
    if df is None or len(df) < 100:
        return jsonify({"error": "Insufficient data"}), 500
    prices = df['Close_INR'].values
    if Sequential is None: return statistical_forecast(prices)
    
    if lstm_model_cache['model'] is None:
        model, scaler = train_lstm_model(prices)
        lstm_model_cache.update({'model': model, 'scaler': scaler})
    
    last_sequence = prices[-60:]
    last_sequence_scaled = lstm_model_cache['scaler'].transform(last_sequence.reshape(-1, 1)).flatten()
    forecast_prices = forecast_lstm(lstm_model_cache['model'], lstm_model_cache['scaler'], last_sequence_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#222')
    ax.set_facecolor('#222')
    ax.plot(range(-90, 0), prices[-90:], color='#8b1e2d')
    ax.plot(range(0, 30), forecast_prices, color='#4CAF50', linestyle='--')
    ax.tick_params(colors='white')
    img_str = plot_to_base64(fig)
    return jsonify({"image": img_str, "trend": "Upward" if forecast_prices[-1] > prices[-1] else "Downward"})

def statistical_forecast(prices):
    z = np.polyfit(range(90), prices[-90:], 2)
    p = np.poly1d(z)
    forecast_prices = p(range(90, 120))
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#222')
    ax.plot(forecast_prices)
    img_str = plot_to_base64(fig)
    return jsonify({"image": img_str, "trend": "Statistical"})

@app.route('/api/inflation')
def inflation_info():
    return jsonify({"title": "Inflation Impact", "summary": "Silver acts as a hedge."})

@app.route('/api/availability')
def availability_info():
    return jsonify({"title": "Global Supply", "key_points": ["India imports 70%"]})

@app.route('/api/renewable-energy')
def renewable_energy_info():
    return jsonify({"title": "Renewable Energy", "solar_impact": "Solar panels use silver paste."})

@app.route('/api/geopolitical')
def geopolitical_info():
    return jsonify({"title": "Geopolitical Factors", "major_factors": [{"factor": "US-China Trade", "impact": "High"}]})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
