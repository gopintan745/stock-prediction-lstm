"""
Prediction service for the Stock Prediction LSTM application.
Handles data fetching, preprocessing, and making predictions with the trained model.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from data.pipeline import FEATURE_COLUMNS, TARGET_COLUMN, add_technical_indicators, make_sequences
from models.lstm import StockLSTM


class PredictionService:
    """Service for making stock predictions using the trained LSTM model."""
    
    def __init__(
        self,
        model: StockLSTM,
        scaler_x: StandardScaler,
        scaler_y: StandardScaler,
        window: int = 30,
        device: str = "cpu"
    ):
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.window = window
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def fetch_recent_data(self, ticker: str, days: int = 100) -> pd.DataFrame:
        """
        Fetch recent OHLCV data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            repair=True
        )
        
        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'")
        
        df = df.reset_index()
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if c[0] != "" else c[1] for c in df.columns]
        
        # Ensure date column is named "Date"
        if "Date" not in df.columns:
            for col in ["Datetime", "date", "datetime"]:
                if col in df.columns:
                    df = df.rename(columns={col: "Date"})
                    break
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and prepare features for prediction.
        
        Args:
            df: Raw OHLCV DataFrame
        
        Returns:
            DataFrame with features and target
        """
        return add_technical_indicators(df)
    
    def make_prediction(self, ticker: str) -> Dict[str, Any]:
        """
        Make a prediction for the next day's log return.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with prediction results
        """
        # Fetch recent data (need enough for window + indicators)
        df = self.fetch_recent_data(ticker, days=100)
        
        # Add technical indicators
        df = self.prepare_features(df)
        
        # Check if we have enough data
        if len(df) < self.window:
            raise ValueError(f"Insufficient data: need at least {self.window} days, got {len(df)}")
        
        # Get the last window of features
        features = df[FEATURE_COLUMNS].iloc[-self.window:].values
        
        # Scale features
        features_scaled = self.scaler_x.transform(features)
        
        # Convert to tensor and add batch dimension
        X = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            pred_scaled = self.model(X).cpu().numpy()
        
        # Inverse transform to get actual log return
        pred_log_return = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
        
        # Get the latest close price for context
        latest_close = df["Close"].iloc[-1]
        latest_date = df["Date"].iloc[-1]
        
        # Calculate predicted price (from log return)
        predicted_price = latest_close * np.exp(pred_log_return)
        
        # Direction
        direction = "UP" if pred_log_return > 0 else "DOWN"
        
        return {
            "ticker": ticker,
            "prediction_date": latest_date,
            "latest_close": float(latest_close),
            "predicted_log_return": float(pred_log_return),
            "predicted_price": float(predicted_price),
            "direction": direction,
            "window_size": self.window,
            "features_used": FEATURE_COLUMNS,
        }
    
    def get_historical_predictions(
        self,
        ticker: str,
        test_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get historical predictions for backtesting/visualization.
        
        Args:
            ticker: Stock ticker symbol
            test_days: Number of recent days to predict
        
        Returns:
            List of prediction dictionaries
        """
        # Fetch more data for historical predictions
        df = self.fetch_recent_data(ticker, days=test_days + self.window + 50)
        df = self.prepare_features(df)
        
        predictions = []
        n = len(df)
        
        # We need at least window + 1 rows to make a prediction (window for features, 1 for target)
        # Start from the end and go back test_days
        for i in range(min(test_days, n - self.window)):
            # end_idx is the index of the day we're predicting (the target day)
            # start_idx is the start of the window
            end_idx = n - 1 - i  # target day index
            start_idx = end_idx - self.window  # window start index
            
            if start_idx < 0:
                continue
            
            features = df[FEATURE_COLUMNS].iloc[start_idx:end_idx].values
            
            # Skip if we don't have enough features
            if len(features) != self.window:
                continue
            
            features_scaled = self.scaler_x.transform(features)
            X = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_scaled = self.model(X).cpu().numpy()
            
            pred_log_return = self.scaler_y.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).ravel()[0]
            
            actual_log_return = df[TARGET_COLUMN].iloc[end_idx]
            latest_close = df["Close"].iloc[end_idx]
            pred_date = df["Date"].iloc[end_idx]
            
            predictions.append({
                "date": pred_date,
                "actual_log_return": float(actual_log_return),
                "predicted_log_return": float(pred_log_return),
                "close_price": float(latest_close),
                "direction_correct": (pred_log_return > 0) == (actual_log_return > 0)
            })
        
        return predictions


def create_prediction_service(
    model: StockLSTM,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    window: int = 30,
    device: str = "cpu"
) -> PredictionService:
    """Factory function to create a PredictionService instance."""
    return PredictionService(model, scaler_x, scaler_y, window, device)
