"""
Gradio application for Stock Price Prediction using LSTM.

Run with: python -m app.gradio_app
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import torch

from app.model_loader import load_artifacts, get_default_config
from app.prediction_service import create_prediction_service
from data.pipeline import FEATURE_COLUMNS


# Global cache for loaded models
_model_cache = {}


def load_model_cached(ticker: str, use_hf_hub: bool, hf_repo_id: str):
    """Load model and artifacts with caching."""
    cache_key = (ticker, use_hf_hub, hf_repo_id)
    if cache_key not in _model_cache:
        model, scaler_x, scaler_y, config = load_artifacts(
            ticker, 
            use_hf_hub=use_hf_hub, 
            hf_repo_id=hf_repo_id
        )
        _model_cache[cache_key] = (model, scaler_x, scaler_y, config)
    return _model_cache[cache_key]


def make_prediction(ticker, use_hf_hub, hf_repo_id, device):
    """Make a prediction for the given ticker."""
    try:
        model, scaler_x, scaler_y, config = load_model_cached(ticker, use_hf_hub, hf_repo_id)
        prediction_service = create_prediction_service(
            model, scaler_x, scaler_y,
            window=config.get("window", 30),
            device=device
        )
        result = prediction_service.make_prediction(ticker)
        
        # Format output
        direction_emoji = "🟢" if result['direction'] == "UP" else "🔴"
        pred_pct = result['predicted_log_return'] * 100
        
        output = f"""
## 📊 Prediction Results for {ticker}

| Metric | Value |
|--------|-------|
| **Latest Close** | ${result['latest_close']:.2f} |
| **Predicted Log Return** | {result['predicted_log_return']:.6f} |
| **Predicted Price** | ${result['predicted_price']:.2f} |
| **Direction** | {direction_emoji} {result['direction']} |
| **Change** | {((result['predicted_price'] - result['latest_close']) / result['latest_close'] * 100):.2f}% |
| **Prediction Date** | {result['prediction_date']} |

### 📈 Interpretation
"""
        if abs(pred_pct) < 0.1:
            output += f"📍 The model predicts a **nearly flat** day ({pred_pct:.3f}% log return)."
        elif pred_pct > 0:
            output += f"📈 The model predicts an **UP** day with ~{pred_pct:.3f}% log return."
        else:
            output += f"📉 The model predicts a **DOWN** day with ~{pred_pct:.3f}% log return."
        
        output += """

---
⚠️ **Disclaimer:** This prediction is based on a machine learning model trained on historical data. It is NOT financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.
"""
        return output
    except Exception as e:
        return f"❌ Error making prediction: {str(e)}"


def run_backtest(ticker, test_days, use_hf_hub, hf_repo_id, device):
    """Run backtest and return plot and metrics."""
    try:
        model, scaler_x, scaler_y, config = load_model_cached(ticker, use_hf_hub, hf_repo_id)
        prediction_service = create_prediction_service(
            model, scaler_x, scaler_y,
            window=config.get("window", 30),
            device=device
        )
        historical = prediction_service.get_historical_predictions(ticker, test_days=test_days)
        
        if not historical:
            return "⚠️ Not enough data for backtest.", None
        
        df_hist = pd.DataFrame(historical)
        
        # Directional accuracy
        dir_acc = df_hist['direction_correct'].mean() * 100
        
        # Create plot
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
        
        # Metrics text
        metrics_text = f"""
## 📊 Backtest Results for {ticker} (Last {test_days} Days)

| Metric | Value |
|--------|-------|
| **Directional Accuracy** | {dir_acc:.1f}% |
| **Total Predictions** | {len(df_hist)} |
| **Correct Directions** | {df_hist['direction_correct'].sum()} |
"""
        
        return metrics_text, fig
    except Exception as e:
        return f"❌ Error running backtest: {str(e)}", None


def get_data_preview(ticker, use_hf_hub, hf_repo_id, device):
    """Get recent data preview."""
    try:
        model, scaler_x, scaler_y, config = load_model_cached(ticker, use_hf_hub, hf_repo_id)
        prediction_service = create_prediction_service(
            model, scaler_x, scaler_y,
            window=config.get("window", 30),
            device=device
        )
        df_raw = prediction_service.fetch_recent_data(ticker, days=30)
        df_features = prediction_service.prepare_features(df_raw)
        
        # Show last 10 rows of key columns
        display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD', 'log_return']
        available_cols = [c for c in display_cols if c in df_features.columns]
        
        df_display = df_features[available_cols].tail(10).copy()
        
        # Format for display
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"${x:.2f}")
        if 'Volume' in df_display.columns:
            df_display['Volume'] = df_display['Volume'].apply(lambda x: f"{x:,.0f}")
        if 'RSI_14' in df_display.columns:
            df_display['RSI_14'] = df_display['RSI_14'].apply(lambda x: f"{x:.2f}")
        if 'MACD' in df_display.columns:
            df_display['MACD'] = df_display['MACD'].apply(lambda x: f"{x:.4f}")
        if 'log_return' in df_display.columns:
            df_display['log_return'] = df_display['log_return'].apply(lambda x: f"{x:.6f}")
        
        return df_display
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def get_model_info(ticker, use_hf_hub, hf_repo_id):
    """Get model configuration info."""
    try:
        model, scaler_x, scaler_y, config = load_model_cached(ticker, use_hf_hub, hf_repo_id)
        
        info = f"""
## 📋 Model Configuration

| Parameter | Value |
|-----------|-------|
| **Ticker** | {ticker} |
| **Window Size** | {config.get('window', 30)} days |
| **Hidden Size** | {config.get('hidden_size', 250)} |
| **Layers** | {config.get('num_layers', 3)} |
| **Dropout** | {config.get('dropout', 0.09):.3f} |
| **Learning Rate** | {config.get('learning_rate', 1.09e-5):.2e} |
| **Batch Size** | {config.get('batch_size', 16)} |
| **Optimizer** | {config.get('optimizer', 'adamw')} |
| **Features** | {len(FEATURE_COLUMNS)} technical indicators |
"""
        return info
    except Exception as e:
        return f"❌ Error loading model info: {str(e)}"


def get_features_list():
    """Get list of features used."""
    features_md = "## 🔧 Features Used\n\n"
    for i, feat in enumerate(FEATURE_COLUMNS, 1):
        features_md += f"{i}. {feat}\n"
    return features_md


# Create Gradio interface
with gr.Blocks(title="Stock Prediction LSTM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📈 Stock Price Prediction - LSTM")
    gr.Markdown("Next-day log return prediction using LSTM with technical indicators")
    
    with gr.Sidebar():
        gr.Markdown("## ⚙️ Settings")
        
        ticker = gr.Textbox(
            label="Stock Ticker",
            value="AAPL",
            placeholder="Enter ticker (e.g., AAPL, GOOGL, MSFT)"
        )
        
        use_hf_hub = gr.Checkbox(
            label="Load from Hugging Face Hub",
            value=True,
            info="Load model artifacts from HF Hub instead of local files"
        )
        
        hf_repo_id = gr.Textbox(
            label="HF Repository ID",
            value="RousingSea7309/stock-prediction-lstm",
            visible=True
        )
        
        device = gr.Radio(
            choices=["cpu", "cuda"],
            value="cpu",
            label="Device",
            info="Use CUDA if available for faster inference"
        )
        
        # Update HF repo visibility
        def toggle_hf_repo(use_hub):
            return gr.update(visible=use_hub)
        
        use_hf_hub.change(toggle_hf_repo, inputs=use_hf_hub, outputs=hf_repo_id)
        
        gr.Markdown("---")
        
        # Model info
        model_info_btn = gr.Button("🔄 Load Model Info", variant="secondary")
        model_info_md = gr.Markdown()
        
        features_md = gr.Markdown(get_features_list())
    
    with gr.Tabs():
        with gr.TabItem("🔮 Prediction"):
            predict_btn = gr.Button("🔮 Make Prediction", variant="primary", size="lg")
            prediction_output = gr.Markdown()
            
            predict_btn.click(
                fn=make_prediction,
                inputs=[ticker, use_hf_hub, hf_repo_id, device],
                outputs=prediction_output
            )
        
        with gr.TabItem("📊 Backtest"):
            test_days = gr.Slider(
                minimum=5, maximum=60, value=30, step=5,
                label="Test Days",
                info="Number of recent days to backtest"
            )
            backtest_btn = gr.Button("📈 Run Backtest", variant="primary")
            backtest_metrics = gr.Markdown()
            backtest_plot = gr.Plot()
            
            backtest_btn.click(
                fn=run_backtest,
                inputs=[ticker, test_days, use_hf_hub, hf_repo_id, device],
                outputs=[backtest_metrics, backtest_plot]
            )
        
        with gr.TabItem("📋 Data Preview"):
            preview_btn = gr.Button("🔄 Refresh Data", variant="secondary")
            data_table = gr.Dataframe(
                label="Recent Data (Last 10 Days)",
                wrap=True
            )
            
            preview_btn.click(
                fn=get_data_preview,
                inputs=[ticker, use_hf_hub, hf_repo_id, device],
                outputs=data_table
            )
        
        with gr.TabItem("ℹ️ Model Info"):
            model_info_md = gr.Markdown()
            
            model_info_btn.click(
                fn=get_model_info,
                inputs=[ticker, use_hf_hub, hf_repo_id],
                outputs=model_info_md
            )
    
    # Load model info on startup
    demo.load(
        fn=get_model_info,
        inputs=[ticker, use_hf_hub, hf_repo_id],
        outputs=model_info_md
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
