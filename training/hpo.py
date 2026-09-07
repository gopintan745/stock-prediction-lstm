from argparse import ArgumentParser
import optuna
import torch
from data.pipeline import FEATURE_COLUMNS, load_dataset
from training.loop import create_dataloaders, train_model, select_optimizer, set_seed
from models.lstm import StockLSTM


def objective(trial, device='cpu', num_epochs=50):
    # Hyperparameters to tune
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    window = trial.suggest_categorical('window', [30, 60, 90])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw'])

    # Load dataset
    train_ds, val_ds, test_ds, _, _ = load_dataset("AAPL", window=window)

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)

    # Initialize model
    input_size = len(FEATURE_COLUMNS)
    model = StockLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = select_optimizer(model, optimizer_name=optimizer_name, learning_rate=learning_rate)

    # Train the model
    _, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, seed=42, device=device)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_loss


def run_optuna_study(n_trials=50, device='cpu', num_epochs=50):
    study = optuna.create_study(direction='minimize', 
                                study_name='lstm_hyperparameter_optimization',
                                storage='sqlite:///optuna_study.db', 
                                load_if_exists=True, 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5),
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, device=device, num_epochs=num_epochs), n_trials=n_trials, n_jobs=1)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def main():
    parser = ArgumentParser(description="Hyperparameter Optimization for LSTM Stock Prediction")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for Optuna study")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the training on (cpu or cuda)")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training in each trial")
    args = parser.parse_args()

    run_optuna_study(n_trials=args.n_trials, device=args.device, num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()