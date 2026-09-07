import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score, MeanAbsolutePercentageError

def directional_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the directional accuracy of predictions.
    Directional accuracy is the percentage of times the model correctly predicts the direction of change.
    """
    # Ensure preds and targets are 1D tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Calculate the direction of change
    pred_direction = torch.sign(preds[1:] - preds[:-1])
    target_direction = torch.sign(targets[1:] - targets[:-1])

    # Calculate directional accuracy
    correct_directions = (pred_direction == target_direction).sum().item()
    total_directions = len(pred_direction)

    return correct_directions / total_directions if total_directions > 0 else 0.0


def evaluate(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Evaluate various regression metrics for the predictions.
    Returns a dictionary containing MSE, MAE, R2, MAPE, and directional accuracy.
    """
    rmse = MeanSquaredError(squared=False)(preds, targets).item()
    mae = MeanAbsoluteError()(preds, targets).item()
    r2 = R2Score()(preds, targets).item() if len(preds) > 1 else float('nan')  # R2 is not defined for a single sample
    mape = MeanAbsolutePercentageError()(preds, targets).item()
    dir_acc = directional_accuracy(preds, targets)

    return {
        "MSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "Directional Accuracy": dir_acc
    }


def naive_baseline_preds(targets: torch.Tensor) -> torch.Tensor:
    """
    Generate naive baseline predictions by using the previous value as the prediction for the next time step.
    The first prediction is set to be equal to the first target value.
    """
    preds = torch.zeros_like(targets)
    preds[0] = targets[0]  # First prediction is the same as the first target
    preds[1:] = targets[:-1]  # Subsequent predictions are the previous target values
    return preds