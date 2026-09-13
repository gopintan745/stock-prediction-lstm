"""
Model loader utility for the Stock Prediction LSTM application.
Handles loading the trained model, scalers, and configuration from local files or Hugging Face Hub.
"""

import json
import pickle
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
from sklearn.preprocessing import StandardScaler

from models.lstm import StockLSTM

# Try to import huggingface_hub for remote loading
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def load_artifacts(
    ticker: str = "AAPL", 
    artifacts_dir: Optional[str] = None,
    use_hf_hub: bool = False,
    hf_repo_id: str = "RousingSea7309/stock-prediction-lstm"
) -> Tuple[StockLSTM, StandardScaler, StandardScaler, Dict[str, Any]]:
    """
    Load the trained model, scalers, and configuration for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        artifacts_dir: Directory containing model artifacts. Defaults to project root.
        use_hf_hub: If True, download artifacts from Hugging Face Hub
        hf_repo_id: Hugging Face repository ID (used if use_hf_hub=True)
    
    Returns:
        Tuple of (model, scaler_x, scaler_y, config)
    """
    if use_hf_hub:
        return _load_from_hf_hub(ticker, hf_repo_id)
    
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).parent.parent
    else:
        artifacts_dir = Path(artifacts_dir)
    
    return _load_from_local(ticker, artifacts_dir)


def _load_from_local(
    ticker: str, 
    artifacts_dir: Path
) -> Tuple[StockLSTM, StandardScaler, StandardScaler, Dict[str, Any]]:
    """Load artifacts from local filesystem."""
    
    # Load configuration
    config_path = artifacts_dir / f"config_{ticker}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load scalers
    scalers_path = artifacts_dir / f"scalers_{ticker}.pkl"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    scaler_x = scalers["scaler_x"]
    scaler_y = scalers["scaler_y"]
    
    # Load model with correct architecture from config/optuna
    # Use the best hyperparameters from the Optuna study
    model = StockLSTM(
        input_size=12,  # len(FEATURE_COLUMNS) from data.pipeline
        hidden_size=config.get("hidden_size", 250),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.09229481465680286),
    )
    
    model_path = artifacts_dir / f"model_{ticker}.pt"
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, scaler_x, scaler_y, config


def _load_from_hf_hub(
    ticker: str,
    hf_repo_id: str
) -> Tuple[StockLSTM, StandardScaler, StandardScaler, Dict[str, Any]]:
    """Load artifacts from Hugging Face Hub."""
    
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to load from HF Hub. "
            "Install with: pip install huggingface_hub"
        )
    
    # Download files from HF Hub
    config_path = hf_hub_download(repo_id=hf_repo_id, filename=f"config_{ticker}.json")
    scalers_path = hf_hub_download(repo_id=hf_repo_id, filename=f"scalers_{ticker}.pkl")
    model_path = hf_hub_download(repo_id=hf_repo_id, filename=f"model_{ticker}.pt")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load scalers
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    scaler_x = scalers["scaler_x"]
    scaler_y = scalers["scaler_y"]
    
    # Load model
    model = StockLSTM(
        input_size=12,
        hidden_size=config.get("hidden_size", 250),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.09229481465680286),
    )
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, scaler_x, scaler_y, config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration (from Optuna best params)."""
    return {
        "hidden_size": 250,
        "num_layers": 3,
        "dropout": 0.09229481465680286,
        "learning_rate": 1.0906196931243558e-05,
        "window": 30,
        "batch_size": 16,
        "optimizer": "adamw",
    }
