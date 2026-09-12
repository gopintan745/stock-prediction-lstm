"""
Command-line interface for the Stock Prediction LSTM application.
Provides quick predictions without the web UI.
"""

import argparse
import sys
from pathlib import Path

import torch

from app.model_loader import load_artifacts
from app.prediction_service import create_prediction_service


def predict(ticker: str = "AAPL", device: str = "cpu") -> None:
    """Make a single prediction for a ticker."""
    print(f"Loading model for {ticker}...")
    
    try:
        model, scaler_x, scaler_y, config = load_artifacts(ticker)
        service = create_prediction_service(
            model, scaler_x, scaler_y,
            window=config.get("window", 30),
            device=device
        )
        
        print("Fetching data and making prediction...")
        result = service.make_prediction(ticker)
        
        print("\n" + "=" * 50)
        print(f"PREDICTION FOR {ticker}")
        print("=" * 50)
        print(f"Latest Close:       ${result['latest_close']:.2f}")
        print(f"Predicted Log Return: {result['predicted_log_return']:.6f}")
        print(f"Predicted Price:    ${result['predicted_price']:.2f}")
        print(f"Direction:          {result['direction']}")
        print(f"Prediction Date:    {result['prediction_date']}")
        print(f"Window Size:        {result['window_size']} days")
        print("=" * 50)
        
        # Interpretation
        pred_return_pct = result['predicted_log_return'] * 100
        if abs(pred_return_pct) < 0.1:
            print(f"\n📍 The model predicts a NEARLY FLAT day ({pred_return_pct:.3f}% log return).")
        elif pred_return_pct > 0:
            print(f"\n📈 The model predicts an UP day with ~{pred_return_pct:.3f}% log return.")
        else:
            print(f"\n📉 The model predicts a DOWN day with ~{pred_return_pct:.3f}% log return.")
        
        print("\n⚠️  DISCLAIMER: This prediction is based on a machine learning model trained on historical data.")
        print("   It is NOT financial advice. Past performance does not guarantee future results.")
        print("   Always do your own research before making investment decisions.")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def backtest(ticker: str = "AAPL", days: int = 30, device: str = "cpu") -> None:
    """Run backtest for a ticker."""
    print(f"Loading model for {ticker}...")
    
    try:
        model, scaler_x, scaler_y, config = load_artifacts(ticker)
        service = create_prediction_service(
            model, scaler_x, scaler_y,
            window=config.get("window", 30),
            device=device
        )
        
        print(f"Running backtest for last {days} days...")
        historical = service.get_historical_predictions(ticker, test_days=days)
        
        if not historical:
            print("Not enough data for backtest.")
            return
        
        print("\n" + "=" * 70)
        print(f"BACKTEST RESULTS FOR {ticker} (Last {len(historical)} days)")
        print("=" * 70)
        print(f"{'Date':<12} {'Predicted':>12} {'Actual':>12} {'Correct':>8}")
        print("-" * 70)
        
        correct = 0
        for h in historical:
            is_correct = h['direction_correct']
            if is_correct:
                correct += 1
            status = "✓" if is_correct else "✗"
            print(f"{str(h['date'])[:10]:<12} {h['predicted_log_return']:>12.6f} {h['actual_log_return']:>12.6f} {status:>8}")
        
        accuracy = correct / len(historical) * 100
        print("-" * 70)
        print(f"Directional Accuracy: {accuracy:.1f}% ({correct}/{len(historical)})")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Stock Prediction LSTM - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.cli predict --ticker AAPL
  python -m app.cli predict --ticker GOOGL --device cuda
  python -m app.cli backtest --ticker AAPL --days 30
  python -m app.cli backtest --ticker MSFT --days 60 --device cuda
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make a single prediction")
    predict_parser.add_argument("--ticker", "-t", default="AAPL", help="Stock ticker (default: AAPL)")
    predict_parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], help="Device to use (default: cpu)")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    backtest_parser.add_argument("--ticker", "-t", default="AAPL", help="Stock ticker (default: AAPL)")
    backtest_parser.add_argument("--days", "-n", type=int, default=30, help="Number of days to backtest (default: 30)")
    backtest_parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], help="Device to use (default: cpu)")
    
    args = parser.parse_args()
    
    if args.command == "predict":
        predict(args.ticker.upper(), args.device)
    elif args.command == "backtest":
        backtest(args.ticker.upper(), args.days, args.device)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
