import torch
import optuna
import json
import pickle
from argparse import ArgumentParser
from data.pipeline import FEATURE_COLUMNS, load_dataset
from training.loop import create_dataloaders, train_model, select_optimizer
from models.lstm import StockLSTM
from utils.metrics import evaluate, naive_baseline_preds


def final_training(ticker="AAPL", num_epochs=100, device='cpu'):

    study = optuna.load_study(
        study_name='lstm_hyperparameter_optimization',
        storage='sqlite:///optuna_study.db'
    ) 
    # Load dataset with a fixed window size for final training
    window_size = study.best_params['window'] # You can choose the best window size based on previous HPO results
    train_ds, val_ds, test_ds, scaler_x, scaler_y, _ = load_dataset(ticker=ticker, window_size=window_size)

    # Create dataloaders
    batch_size = study.best_params['batch_size']  # You can choose the best batch size based on previous HPO results
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)

    # Initialize model with the best hyperparameters found during HPO
    input_size = len(FEATURE_COLUMNS)
    hidden_size = study.best_params['hidden_size']  # Example value; replace with the best found value
    num_layers = study.best_params['num_layers']    # Example value; replace with the best found value
    dropout = study.best_params['dropout']      # Example value; replace with the best found value

    model = StockLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer_name = study.best_params['optimizer']  # Example value; replace with the best found value
    learning_rate = study.best_params['learning_rate']     # Example value; replace with the best found value
    optimizer = select_optimizer(model, optimizer_name=optimizer_name, learning_rate=learning_rate)

    # Train the model 
    trained_model, final_val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

    # --- Evaluate on test set (unscale predictions back to real return space) ---
    trained_model.eval()
    with torch.no_grad():
        preds_scaled = model(test_loader).cpu().numpy()

    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    targets = scaler_y.inverse_transform(test_ds.y.reshape(-1, 1)).ravel()

    preds_t = torch.tensor(preds, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    model_metrics = evaluate(preds_t, targets_t)
    baseline_metrics = evaluate(naive_baseline_preds(targets_t), targets_t)


    print(f"Final validation loss after training: {final_val_loss:.4f}")

    print("\nModel metrics (test set, log-return space):")
    print(json.dumps(model_metrics, indent=2))
    print("\nNaive baseline metrics (predict zero return):")
    print(json.dumps(baseline_metrics, indent=2))

    return trained_model, scaler_x, scaler_y


def main():
    parser = ArgumentParser(description="Final training of the LSTM model using best hyperparameters from Optuna study.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for final training.")
    parser.add_argument("--device", type=int, default=32, help="Device for final model training.")
    args = parser.parse_args()

    model, scaler_x, scaler_y = final_training(args.ticker, num_epochs=args.epochs, device=args.device)
    # --- Save artifacts for the Streamlit app ---
    torch.save(model.state_dict(), f"model_{args.ticker}.pt")
    with open(f"scalers_{args.ticker}.pkl", "wb") as f:
        pickle.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, f)
    with open(f"config_{args.ticker}.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSaved model_{args.ticker}.pt, scalers_{args.ticker}.pkl, config_{args.ticker}.json")


if __name__ == "__main__":
    main()

