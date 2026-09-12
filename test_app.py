"""
Test script to verify the Streamlit app components work correctly.
"""
import sys
sys.path.insert(0, '.')

from app.model_loader import load_artifacts, get_default_config
from app.prediction_service import create_prediction_service

def test_model_loading():
    print("Testing model loading...")
    model, scaler_x, scaler_y, config = load_artifacts('AAPL')
    assert model is not None
    assert scaler_x is not None
    assert scaler_y is not None
    assert config is not None
    print(f"  ✓ Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  ✓ Config: {config}")
    return model, scaler_x, scaler_y, config

def test_prediction_service(model, scaler_x, scaler_y, config):
    print("\nTesting prediction service...")
    service = create_prediction_service(
        model, scaler_x, scaler_y,
        window=config.get('window', 30),
        device='cpu'
    )
    
    # Test single prediction
    result = service.make_prediction('AAPL')
    assert 'predicted_log_return' in result
    assert 'predicted_price' in result
    assert 'direction' in result
    print(f"  ✓ Single prediction: {result['direction']} (log_return={result['predicted_log_return']:.6f})")
    
    # Test historical predictions
    historical = service.get_historical_predictions('AAPL', test_days=5)
    print(f"  ✓ Historical predictions: {len(historical)} days")
    if historical:
        for h in historical:
            print(f"    {h['date']}: pred={h['predicted_log_return']:.6f}, actual={h['actual_log_return']:.6f}, correct={h['direction_correct']}")
    
    return service

def test_data_fetching(service):
    print("\nTesting data fetching...")
    df = service.fetch_recent_data('AAPL', days=30)
    assert len(df) > 0
    print(f"  ✓ Fetched {len(df)} rows of raw data")
    
    df_features = service.prepare_features(df)
    assert len(df_features) > 0
    print(f"  ✓ Prepared {len(df_features)} rows with features")
    print(f"    Columns: {df_features.columns.tolist()}")

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Stock Prediction LSTM Application")
    print("=" * 50)
    
    model, scaler_x, scaler_y, config = test_model_loading()
    service = test_prediction_service(model, scaler_x, scaler_y, config)
    test_data_fetching(service)
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
