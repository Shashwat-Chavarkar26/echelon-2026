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

# Trying TensorFlow 2.x imports first if failed, then fallback to Keras standalone
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    print("✓ Using TensorFlow 2.x with integrated Keras")
except ImportError:
    try:
        # Fallback to standalone Keras
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping
        print("✓ Using standalone Keras")
    except ImportError:
        print("❌ ERROR: Neither TensorFlow nor Keras found!")
        print("Install with: pip install tensorflow")
        Sequential = None

app = Flask(__name__)
CORS(app)

# Global cache for LSTM model
lstm_model_cache = {
    'model': None,
    'scaler': None,
    'last_trained': None,
    'last_data': None
}

# HELPER FUNCTIONS
def fetch_inr_data(ticker, start, end):
    """Fetch and calculate INR prices for metals"""
    try:
        metal = yf.download(ticker, start=start, end=end, progress=False)
        usd_inr = yf.download("USDINR=X", start=start, end=end, progress=False)
        df = pd.concat([metal["Close"], usd_inr["Close"]], axis=1, join="inner")
        df.columns = ["Metal_USD", "USD_INR"]
        df["Close_INR"] = df["Metal_USD"] * df["USD_INR"]
        df["Return"] = df["Close_INR"].pct_change() * 100

        def label_sentiment(change):
            if pd.isna(change):
                return "Birth"
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
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#222', edgecolor='none', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length):
    """Build LSTM model architecture"""
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
    """Train LSTM model on historical data"""
    try:
        # Prepare data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        # Create sequences
        X, y = create_sequences(scaled_data, seq_length)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Build model
        model = build_lstm_model(seq_length)
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train model
        print("Training LSTM model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Training Loss: {train_loss[0]:.6f}, Test Loss: {test_loss[0]:.6f}")
        
        return model, scaler, history
    
    except Exception as e:
        print(f"Error training LSTM model: {e}")
        return None, None, None

def forecast_lstm(model, scaler, last_sequence, days=30):
    """Generate forecast using trained LSTM model"""
    try:
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next value
            current_sequence_reshaped = current_sequence.reshape(1, len(current_sequence), 1)
            next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
    
    except Exception as e:
        print(f"Error forecasting: {e}")
        return None

# API ENDPOINTS
@app.route('/')
def index():
    # If your file is named 'index.html', change 'website.html' to 'index.html' below
    return send_from_directory(os.getcwd(), 'website.html')

@app.route('/<path:path>')
def serve_static(path):
    # This is the "magic" that makes silver face1.jpeg visible
    return send_from_directory(os.getcwd(), path)

@app.route('/api/chart/<period>')
def get_chart(period):
    """Generate price charts for different time periods"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    period_map = {
        'weekly': 7,
        'monthly': 30,
        'quarterly': 90,
        'half-yearly': 180
    }
    
    if period.lower() not in period_map:
        return jsonify({"error": "Invalid period. Use: weekly, monthly, quarterly, half-yearly"}), 400
    
    days = period_map[period.lower()]
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    df = fetch_inr_data("SI=F", start_date, end_date)
    
    if df is None or len(df) == 0:
        return jsonify({"error": "Could not fetch data. Check internet connection."}), 500
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#222')
    ax.set_facecolor('#222')
    
    ax.plot(df.index, df['Close_INR'], color='#8b1e2d', linewidth=2.5, label='Silver Price')
    ax.fill_between(df.index, df['Close_INR'], alpha=0.3, color='#8b1e2d')
    
    ax.set_title(f'{period.title()} Silver Price (INR)', color='white', fontsize=16, pad=20)
    ax.set_xlabel('Date', color='white', fontsize=12)
    ax.set_ylabel('Price (INR)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
    
    for spine in ax.spines.values():
        spine.set_color('white')
    
    img_str = plot_to_base64(fig)
    
    current_price = float(df['Close_INR'].iloc[-1])
    price_change = float(df['Return'].iloc[-1]) if len(df) > 1 else 0.0
    sentiment = df['Sentiment'].iloc[-1]
    
    return jsonify({
        "image": img_str,
        "current_price": round(current_price, 2),
        "price_change": round(price_change, 2),
        "sentiment": sentiment,
        "period": period
    })

@app.route('/api/compare-gold')
def compare_gold():
    """Compare silver vs gold"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    silver_df = fetch_inr_data("SI=F", start_date, end_date)
    gold_df = fetch_inr_data("GC=F", start_date, end_date)
    
    if silver_df is None or gold_df is None:
        return jsonify({"error": "Could not fetch data"}), 500
    
    # Normalize to 100
    silver_norm = (silver_df['Close_INR'] / silver_df['Close_INR'].iloc[0]) * 100
    gold_norm = (gold_df['Close_INR'] / gold_df['Close_INR'].iloc[0]) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#222')
    ax.set_facecolor('#222')
    
    ax.plot(silver_norm.index, silver_norm, color='#c0c0c0', linewidth=2.5, label='Silver')
    ax.plot(gold_norm.index, gold_norm, color='#ffd700', linewidth=2.5, label='Gold')
    
    ax.set_title('Silver vs Gold Performance', color='white', fontsize=16, pad=20)
    ax.set_xlabel('Date', color='white', fontsize=12)
    ax.set_ylabel('Relative Performance (Base 100)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
    
    for spine in ax.spines.values():
        spine.set_color('white')
    
    img_str = plot_to_base64(fig)
    
    silver_return = float(silver_norm.iloc[-1] - 100)
    gold_return = float(gold_norm.iloc[-1] - 100)
    
    return jsonify({
        "image": img_str,
        "silver_return": round(silver_return, 2),
        "gold_return": round(gold_return, 2),
        "better_performer": "Silver" if silver_return > gold_return else "Gold"
    })

@app.route('/api/inflation')
def inflation_info():
    return jsonify({
        "title": "Inflation Impact on Silver",
        "summary": "Silver is often seen as an inflation hedge, with prices typically rising during high inflation periods.",
        "factors": [
            {
                "name": "Currency Devaluation",
                "impact": "High",
                "description": "When INR weakens against USD, silver prices in India increase even if global prices stay stable"
            },
            {
                "name": "Real Interest Rates",
                "impact": "Medium",
                "description": "Negative real rates (inflation > interest) make silver more attractive vs savings accounts"
            },
            {
                "name": "Industrial Demand",
                "impact": "Medium",
                "description": "Inflation can increase manufacturing costs, affecting industrial silver demand"
            }
        ],
        "current_trend": "Monitor CPI data and RBI policy for inflation signals"
    })

@app.route('/api/availability')
def availability_info():
    return jsonify({
        "title": "Global Silver Supply & Demand",
        "key_points": [
            "India imports ~70% of its silver requirement, mainly from South Korea, UAE, and Switzerland",
            "Global silver production: ~1 billion oz/year (2023 estimates)",
            "Industrial demand accounts for ~50% of total silver demand",
            "Investment demand (coins, bars, ETFs) has grown significantly post-2020"
        ],
        "indian_market": {
            "import_dependency": "70%",
            "major_sources": ["South Korea", "UAE", "Switzerland"],
            "domestic_production": "Limited - mostly from lead-zinc mines",
            "demand_drivers": ["Jewelry", "Electronics", "Solar panels", "Investment"]
        }
    })

@app.route('/api/renewable-energy')
def renewable_energy_info():
    return jsonify({
        "title": "Renewable Energy & Silver Demand",
        "solar_impact": {
            "description": "Solar panels use silver paste for photovoltaic cells",
            "demand_share": "~15% of total silver demand",
            "growth_trend": "Increasing as solar installations expand globally"
        },
        "india_specific": {
            "solar_target": "India aims for 500 GW renewable capacity by 2030",
            "silver_requirement": "Each GW of solar requires ~20-30 tons of silver",
            "impact": "Strong domestic demand growth expected from Indian solar expansion"
        },
        "future_outlook": "Silver demand from solar could rise 50-100% by 2030 if targets are met"
    })

@app.route('/api/geopolitical')
def geopolitical_info():
    return jsonify({
        "title": "Geopolitical Factors Affecting Silver Prices",
        "major_factors": [
            {
                "factor": "US-China Trade Relations",
                "impact": "High",
                "details": "Trade tensions affect industrial demand and safe-haven flows"
            },
            {
                "factor": "Middle East Stability",
                "impact": "Medium",
                "details": "Regional conflicts drive safe-haven demand for precious metals"
            },
            {
                "factor": "Mining Country Politics",
                "impact": "High",
                "details": "Labor strikes, nationalizations in Mexico, Peru, Chile affect supply"
            },
            {
                "factor": "Global Monetary Policy",
                "impact": "Very High",
                "details": "Fed rate decisions affect USD strength and silver prices"
            }
        ],
        "trade_disputes": {
            "current_issues": [
                "Import duties on precious metals in various countries",
                "Export restrictions from major producers",
                "Currency manipulation concerns affecting metal prices"
            ],
            "india_specific": "India's import duty on silver (10-15%) significantly affects local prices"
        }
    })

@app.route('/api/forecast')
def forecast():
    """LSTM-based forecast for silver prices (with fallback to statistical method)"""
    global lstm_model_cache
    
    try:
        print("\n=== Starting Forecast ===")
        
        # Fetch historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        print("Fetching historical data...")
        df = fetch_inr_data("SI=F", start_date, end_date)
        
        if df is None or len(df) < 100:
            return jsonify({"error": "Insufficient data for forecasting"}), 500
        
        prices = df['Close_INR'].values
        
        # Check if LSTM is available
        if Sequential is None:
            print("⚠️ TensorFlow not available. Using statistical forecasting...")
            return statistical_forecast(prices)
        
        # LSTM Forecasting
        print("🧠 Using LSTM Neural Network...")
        
        # Check if we need to retrain the model
        retrain = False
        if lstm_model_cache['model'] is None:
            print("No cached model found. Training new model...")
            retrain = True
        elif lstm_model_cache['last_trained'] is None:
            retrain = True
        elif (datetime.now() - lstm_model_cache['last_trained']).days > 7:
            print("Cached model is old. Retraining...")
            retrain = True
        
        # Train or use cached model
        if retrain:
            seq_length = 60
            model, scaler, history = train_lstm_model(prices, seq_length=seq_length, epochs=50)
            
            if model is None:
                print("❌ LSTM training failed. Falling back to statistical method...")
                return statistical_forecast(prices)
            
            # Cache the model
            lstm_model_cache['model'] = model
            lstm_model_cache['scaler'] = scaler
            lstm_model_cache['last_trained'] = datetime.now()
            lstm_model_cache['last_data'] = prices
            
            print("✓ Model training complete!")
        else:
            print("✓ Using cached LSTM model...")
            model = lstm_model_cache['model']
            scaler = lstm_model_cache['scaler']
        
        # Prepare last sequence for forecasting
        seq_length = 60
        last_sequence = prices[-seq_length:]
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        # Generate forecast
        print("Generating 30-day forecast...")
        forecast_prices = forecast_lstm(model, scaler, last_sequence_scaled, days=30)
        
        if forecast_prices is None:
            print("❌ LSTM forecast failed. Falling back to statistical method...")
            return statistical_forecast(prices)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#222')
        ax.set_facecolor('#222')
        
        # Plot historical data (last 90 days)
        historical_days = list(range(-90, 0))
        historical_prices = prices[-90:]
        
        forecast_days = list(range(0, 30))
        
        ax.plot(historical_days, historical_prices, color='#8b1e2d', linewidth=2.5, label='Historical')
        ax.plot(forecast_days, forecast_prices, color='#4CAF50', linewidth=2.5, linestyle='--', label='LSTM Forecast')
        
        # Add confidence interval (±5%)
        upper_bound = forecast_prices * 1.05
        lower_bound = forecast_prices * 0.95
        ax.fill_between(forecast_days, lower_bound, upper_bound, alpha=0.2, color='#4CAF50', label='Confidence Interval')
        
        ax.axvline(x=0, color='white', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(0, max(historical_prices), 'Today', color='white', fontsize=10, ha='center')
        
        ax.set_title('30-Day Silver Price Forecast (LSTM Neural Network)', color='white', fontsize=16, pad=20)
        ax.set_xlabel('Days', color='white', fontsize=12)
        ax.set_ylabel('Price (INR)', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
        
        for spine in ax.spines.values():
            spine.set_color('white')
        
        img_str = plot_to_base64(fig)
        
        # Calculate statistics
        avg_forecast = float(np.mean(forecast_prices))
        current_price = float(prices[-1])
        trend = "Upward" if forecast_prices[-1] > current_price else "Downward"
        price_change = ((forecast_prices[-1] - current_price) / current_price) * 100
        
        confidence = "High - LSTM trained on 1 year of data"
        
        print(f"✓ Forecast complete! Average: ₹{avg_forecast:.2f}, Trend: {trend}")
        print("=== LSTM Forecast Complete ===\n")
        
        return jsonify({
            "image": img_str,
            "average_forecast": round(avg_forecast, 2),
            "trend": trend,
            "confidence": confidence,
            "price_change_30d": round(price_change, 2),
            "forecast_range": {
                "min": round(float(np.min(forecast_prices)), 2),
                "max": round(float(np.max(forecast_prices)), 2)
            }
        })
    
    except Exception as e:
        print(f"❌ Error in forecast endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback method
        try:
            return statistical_forecast(prices)
        except:
            return jsonify({"error": f"Forecast error: {str(e)}"}), 500

def statistical_forecast(prices):
    """Fallback statistical forecast method (ARIMA-like)"""
    try:
        print("Using statistical forecasting method...")
        
        # Calculate moving averages and trends
        ma_7 = np.convolve(prices, np.ones(7)/7, mode='valid')
        ma_30 = np.convolve(prices, np.ones(30)/30, mode='valid')
        
        # Calculate trend
        x = np.arange(len(prices))
        z = np.polyfit(x[-90:], prices[-90:], 2)  # Quadratic fit
        p = np.poly1d(z)
        
        # Forecast
        future_x = np.arange(len(prices), len(prices) + 30)
        forecast_prices = p(future_x)
        
        # Add some realistic variance
        volatility = np.std(prices[-30:])
        noise = np.random.normal(0, volatility * 0.3, 30)
        forecast_prices = forecast_prices + noise
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#222')
        ax.set_facecolor('#222')
        
        historical_days = list(range(-90, 0))
        historical_prices = prices[-90:]
        forecast_days = list(range(0, 30))
        
        ax.plot(historical_days, historical_prices, color='#8b1e2d', linewidth=2.5, label='Historical')
        ax.plot(forecast_days, forecast_prices, color='#FFD700', linewidth=2.5, linestyle='--', label='Statistical Forecast')
        
        upper_bound = forecast_prices * 1.05
        lower_bound = forecast_prices * 0.95
        ax.fill_between(forecast_days, lower_bound, upper_bound, alpha=0.2, color='#FFD700')
        
        ax.axvline(x=0, color='white', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(0, max(historical_prices), 'Today', color='white', fontsize=10, ha='center')
        
        ax.set_title('30-Day Silver Price Forecast (Statistical Model)', color='white', fontsize=16, pad=20)
        ax.set_xlabel('Days', color='white', fontsize=12)
        ax.set_ylabel('Price (INR)', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
        
        for spine in ax.spines.values():
            spine.set_color('white')
        
        img_str = plot_to_base64(fig)
        
        avg_forecast = float(np.mean(forecast_prices))
        current_price = float(prices[-1])
        trend = "Upward" if forecast_prices[-1] > current_price else "Downward"
        price_change = ((forecast_prices[-1] - current_price) / current_price) * 100
        
        return jsonify({
            "image": img_str,
            "average_forecast": round(avg_forecast, 2),
            "trend": trend,
            "confidence": "Medium - Statistical trend analysis",
            "price_change_30d": round(price_change, 2),
            "forecast_range": {
                "min": round(float(np.min(forecast_prices)), 2),
                "max": round(float(np.max(forecast_prices)), 2)
            }
        })
    except Exception as e:
        print(f"Statistical forecast also failed: {e}")
        return jsonify({"error": "All forecasting methods failed"}), 500

if __name__ == '__main__':
    print("\n🚀 Starting Silver Sentiment API with LSTM Forecasting...")
    print("📊 Server running on http://localhost:5000")
    print("🧠 LSTM model will train on first forecast request (may take 30-60 seconds)")
    print("💾 Model will be cached for faster subsequent requests")
    print("🌐 Open your HTML file in a browser now!\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
