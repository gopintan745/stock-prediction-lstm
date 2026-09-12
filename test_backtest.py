from app.model_loader import load_artifacts
from app.prediction_service import create_prediction_service

model, scaler_x, scaler_y, config = load_artifacts('AAPL')
service = create_prediction_service(model, scaler_x, scaler_y, window=config.get('window', 30), device='cpu')
historical = service.get_historical_predictions('AAPL', test_days=10)
print(f'Number of historical predictions: {len(historical)}')
print('Historical predictions:')
for h in historical:
    print(f"  {h['date']}: pred={h['predicted_log_return']:.6f}, actual={h['actual_log_return']:.6f}, correct={h['direction_correct']}")
if historical:
    print(f'Directional accuracy: {sum(h["direction_correct"] for h in historical)/len(historical)*100:.1f}%')
