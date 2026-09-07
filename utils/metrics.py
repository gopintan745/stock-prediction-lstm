import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

def directional_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the directional accuracy of predictions.
    Directional accuracy is the percentage of times the model correctly predicts
    the direction (up/down) of the log return for each time step.
    """
    # Ensure preds and targets are 1D tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Compare sign of prediction vs sign of target at each time step
    pred_direction = torch.sign(preds)
    target_direction = torch.sign(targets)

    # Ignore zero targets (no change) for directional accuracy
    mask = target_direction != 0
    if mask.sum() == 0:
        return 0.0

    correct_directions = (pred_direction[mask] == target_direction[mask]).sum().item()
    total_directions = mask.sum().item()

    return correct_directions / total_directions if total_directions > 0 else 0.0


def evaluate(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Evaluate various regression metrics for the predictions.
    Returns a dictionary containing MSE, MAE, R2, SMAPE, and directional accuracy.
    Note: MAPE is problematic for log returns (can be zero/negative), so we use SMAPE instead.
    """
    rmse = MeanSquaredError(squared=False)(preds, targets).item()
    mae = MeanAbsoluteError()(preds, targets).item()
    r2 = R2Score()(preds, targets).item() if len(preds) > 1 else float('nan')
    
    # SMAPE (Symmetric Mean Absolute Percentage Error) - handles zero/negative values better
    # SMAPE = 100% * mean(2 * |pred - target| / (|pred| + |target| + epsilon))
    epsilon = 1e-8
    smape = (2.0 * torch.abs(preds - targets) / (torch.abs(preds) + torch.abs(targets) + epsilon)).mean().item() * 100
    
    dir_acc = directional_accuracy(preds, targets)

    return {
        "MSE": rmse,
        "MAE": mae,
        "R2": r2,
        "SMAPE": smape,
        "Directional Accuracy": dir_acc
    }


def naive_baseline_preds(targets: torch.Tensor) -> torch.Tensor:
    """
    Generate naive baseline predictions by predicting zero return (no change).
    This is a more appropriate baseline for log returns than previous-value persistence.
    """
    return torch.zeros_like(targets)