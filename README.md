---
title: Stock Prediction LSTM
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# Stock Prediction LSTM

A PyTorch-based LSTM model for predicting next-day stock log returns using technical indicators. Deployed as a Hugging Face Space with Streamlit.

## Model

The model is a 3-layer LSTM with 250 hidden units, trained on AAPL stock data from 2015-present using technical indicators (SMA, EMA, RSI, MACD, log returns).

**Model Repository**: [RousingSea7309/stock-prediction-lstm](https://huggingface.co/RousingSea7309/stock-prediction-lstm)

## Features

- Next-day log return prediction
- Technical indicators: SMA, EMA, RSI, MACD
- Interactive Streamlit UI
- Backtesting visualization
- Directional accuracy metrics

## Usage

1. Enter a stock ticker (default: AAPL)
2. Click "Make Prediction" to get next-day prediction
3. Run backtest to see historical performance

## Disclaimer

⚠️ **This is NOT financial advice.** The predictions are based on a machine learning model trained on historical data. Past performance does not guarantee future results. Always do your own research before making investment decisions.
