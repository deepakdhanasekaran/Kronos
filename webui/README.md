# Kronos Crypto Dashboard

Lightweight Flask dashboard for the warm Kronos prediction API. It shows the daily Binance top 30 USDT coins, lets you add extra coins, and refreshes live signals every 10 seconds.

## ✨ Features

- **Top 30 USDT pairs**: Ranked by Binance 24h quote volume
- **Custom coins**: Persisted in a mounted JSON volume
- **Warm backend**: The model is loaded once in the predictor container
- **10-second refresh**: Frontend auto-refreshes the table and chips
- **Focused fields**: Only shows the fields needed for trading review

## 🚀 Quick Start

### Method 1: Start with Python script
```bash
cd webui
python run.py
```

### Method 2: Start with Shell script
```bash
cd webui
chmod +x start.sh
./start.sh
```

### Method 3: Start Flask application directly
```bash
cd webui
python app.py
```

After successful startup, visit `http://localhost:7070`

### Method 4: Full Docker stack

```bash
docker compose up --build
```

## 📋 Usage Steps

1. **Start the stack**: Run Docker Compose or launch the backend and frontend separately
2. **Review the top 30**: The dashboard pulls the daily Binance ranking automatically
3. **Add custom coins**: Enter an extra `USDT` pair in the form and save it
4. **Refresh live**: The table updates every 10 seconds

## 🔧 Prediction Quality Parameters

### Environment Variables

- `KRONOS_BACKEND_URL`: URL of the warm predictor service
- `DASHBOARD_WATCHLIST_PATH`: Path to the persisted custom coin registry
- `DASHBOARD_REFRESH_SECONDS`: Auto-refresh interval
- `DASHBOARD_INTERVAL`: Candle interval sent to the backend
- `DASHBOARD_LOOKBACK`: Number of historical candles used for context
- `DASHBOARD_PRED_LEN`: Prediction horizon
- `DASHBOARD_SAMPLE_COUNT`: Sampling count for the backend model
- `DASHBOARD_CONFIDENCE_SAMPLES`: Direction agreement sampling count

## 📊 Supported Data Formats

### Required Symbols
- Use Binance-style `USDT` pairs such as `BTCUSDT`

## 🤖 Model Support

- **Kronos-small**: 24.7M parameters, balanced performance and speed
- **Kronos-base**: 102.3M parameters, higher quality prediction

## 🖥️ GPU Acceleration Support

- **CPU**: General computing, best compatibility
- **CUDA**: NVIDIA GPU acceleration, best performance
- **MPS**: Apple Silicon GPU acceleration, recommended for Mac users

## ⚠️ Notes

- The dashboard expects `USDT` pairs
- The custom coin registry is persisted in a mounted volume when using Docker
- The first model load may take a while while weights are downloaded

## 🔍 Comparison Analysis

The system automatically provides comparison analysis between prediction results and actual data, including:
- Price difference statistics
- Error analysis
- Prediction quality assessment

## 🛠️ Technical Architecture

- **Backend**: Flask + Python
- **Frontend**: HTML + CSS + JavaScript
- **Charts**: Plotly.js
- **Data processing**: Pandas + NumPy
- **Model**: Hugging Face Transformers

## 📝 Troubleshooting

### Common Issues
1. **Port occupied**: Modify port number in app.py
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Model loading failed**: Check network connection and model ID
4. **Data format error**: Ensure data column names and format are correct

### Log Viewing
Detailed runtime information will be displayed in the console at startup, including model status and error messages.

## 📄 License

This project follows the license terms of the original Kronos project.

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this Web UI!

## 📞 Support

If you have questions, please check:
1. Project documentation
2. GitHub Issues
3. Console error messages
