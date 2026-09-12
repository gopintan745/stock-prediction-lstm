"""
Streamlit application for Stock Price Prediction using LSTM.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import torch

from app.model_loader import load_artifacts, get_default_config
from app.prediction_service import create_prediction_service
from data.pipeline import FEATURE_COLUMNS


# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction - LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_and_artifacts(ticker: str):
    """Load model and artifacts (cached for performance)."""
    return load_artifacts(ticker)


def main():
    # Sidebar
    st.sidebar.title("📈 Stock Prediction LSTM")
    st.sidebar.markdown("---")
    
    # Ticker selection
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter a valid stock ticker (e.g., AAPL, GOOGL, MSFT)").upper()
    
    # Device selection
    device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0, help="Use CUDA if available for faster inference")
    
    # Load model
    try:
        with st.spinner(f"Loading model for {ticker}..."):
            model, scaler_x, scaler_y, config = load_model_and_artifacts(ticker)
            prediction_service = create_prediction_service(
                model, scaler_x, scaler_y,
                window=config.get("window", 30),
                device=device
            )
        st.sidebar.success(f"✅ Model loaded for {ticker}")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {str(e)}")
        st.stop()
    
    # Model info in sidebar
    with st.sidebar.expander("Model Configuration"):
        st.json(config)
    
    # Main content
    st.title(f"📈 {ticker} Stock Price Prediction")
    st.markdown("Next-day log return prediction using LSTM with technical indicators")
    
    # Make prediction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
            with st.spinner("Fetching data and making prediction..."):
                try:
                    result = prediction_service.make_prediction(ticker)
                    
                    # Display results
                    st.success("Prediction completed!")
                    
                    # Metrics row
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric(
                            "Latest Close",
                            f"${result['latest_close']:.2f}",
                            help="Most recent closing price"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Predicted Log Return",
                            f"{result['predicted_log_return']:.6f}",
                            help="Predicted next-day log return"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Predicted Price",
                            f"${result['predicted_price']:.2f}",
                            delta=f"{((result['predicted_price'] - result['latest_close']) / result['latest_close'] * 100):.2f}%",
                            help="Predicted next-day closing price"
                        )
                    
                    with metric_col4:
                        direction_color = "🟢" if result['direction'] == "UP" else "🔴"
                        st.metric(
                            "Direction",
                            f"{direction_color} {result['direction']}",
                            help="Predicted price direction"
                        )
                    
                    # Interpretation
                    st.markdown("---")
                    st.subheader("📊 Interpretation")
                    
                    pred_return_pct = result['predicted_log_return'] * 100
                    if abs(pred_return_pct) < 0.1:
                        st.info(f"📍 The model predicts a **nearly flat** day ({pred_return_pct:.3f}% log return).")
                    elif pred_return_pct > 0:
                        st.success(f"📈 The model predicts an **UP** day with ~{pred_return_pct:.3f}% log return.")
                    else:
                        st.error(f"📉 The model predicts a **DOWN** day with ~{pred_return_pct:.3f}% log return.")
                    
                    # Disclaimer
                    st.warning(
                        "⚠️ **Disclaimer:** This prediction is based on a machine learning model trained on historical data. "
                        "It is NOT financial advice. Past performance does not guarantee future results. "
                        "Always do your own research before making investment decisions."
                    )
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        st.subheader("📋 Model Info")
        st.info(f"""
        **Ticker:** {ticker}
        **Window Size:** {config.get('window', 30)} days
        **Hidden Size:** {config.get('hidden_size', 250)}
        **Layers:** {config.get('num_layers', 3)}
        **Dropout:** {config.get('dropout', 0.09):.3f}
        **Features:** {len(FEATURE_COLUMNS)} technical indicators
        """)
        
        with st.expander("Features Used"):
            for i, feat in enumerate(FEATURE_COLUMNS, 1):
                st.text(f"{i}. {feat}")
    
    # Historical predictions / backtesting
    st.markdown("---")
    st.subheader("📊 Historical Predictions (Backtest)")
    
    col3, col4 = st.columns([3, 1])
    
    with col4:
        test_days = st.slider("Test Days", min_value=5, max_value=60, value=30, help="Number of recent days to backtest")
        show_backtest = st.button("📈 Run Backtest", use_container_width=True)
    
    if show_backtest:
        with st.spinner("Running backtest..."):
            try:
                historical = prediction_service.get_historical_predictions(ticker, test_days=test_days)
                
                if historical:
                    df_hist = pd.DataFrame(historical)
                    
                    # Directional accuracy
                    dir_acc = df_hist['direction_correct'].mean() * 100
                    st.metric("Directional Accuracy", f"{dir_acc:.1f}%")
                    
                    # Plot
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Actual vs Predicted Log Returns", "Close Price"),
                        row_heights=[0.6, 0.4]
                    )
                    
                    # Log returns
                    fig.add_trace(
                        go.Scatter(
                            x=df_hist['date'],
                            y=df_hist['actual_log_return'],
                            name="Actual",
                            line=dict(color='blue', width=1),
                            mode='lines+markers'
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df_hist['date'],
                            y=df_hist['predicted_log_return'],
                            name="Predicted",
                            line=dict(color='red', width=1, dash='dot'),
                            mode='lines+markers'
                        ),
                        row=1, col=1
                    )
                    
                    # Close price
                    fig.add_trace(
                        go.Scatter(
                            x=df_hist['date'],
                            y=df_hist['close_price'],
                            name="Close Price",
                            line=dict(color='green', width=1),
                            mode='lines'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        height=600,
                        title_text=f"{ticker} - Last {test_days} Days Backtest",
                        showlegend=True,
                        hovermode='x unified'
                    )
                    fig.update_yaxes(title_text="Log Return", row=1, col=1)
                    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Show data table
                    with st.expander("View Data"):
                        st.dataframe(df_hist, use_container_width=True)
                else:
                    st.warning("Not enough data for backtest.")
                    
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
    
    # Raw data preview
    st.markdown("---")
    st.subheader("📋 Recent Data Preview")
    
    try:
        df_raw = prediction_service.fetch_recent_data(ticker, days=30)
        df_features = prediction_service.prepare_features(df_raw)
        
        # Show last 10 rows of key columns
        display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD', 'log_return']
        available_cols = [c for c in display_cols if c in df_features.columns]
        
        st.dataframe(
            df_features[available_cols].tail(10).style.format({
                'Open': '${:.2f}',
                'High': '${:.2f}',
                'Low': '${:.2f}',
                'Close': '${:.2f}',
                'Volume': '{:,.0f}',
                'RSI_14': '{:.2f}',
                'MACD': '{:.4f}',
                'log_return': '{:.6f}'
            }),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error loading data preview: {str(e)}")


if __name__ == "__main__":
    main()
