import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

def directional_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of samples where predicted and actual direction (sign) match.
    Most meaningful metric for a trading-relevant evaluation.
 
    Zero predictions (e.g. a constant/no-signal model) are treated as a
    coin-flip -- credited with 0.5 rather than automatically scored as a
    miss. Without this, ANY model that predicts exactly 0 (a degenerate
    but not necessarily "worse" prediction) is unfairly forced to 0%,
    since sign(0) can never equal sign(actual). That makes trivial
    baselines look artificially terrible by comparison to a real model,
    which is a misleading contrast, not a meaningful one.
    """
    pred_dir = torch.sign(preds)
    true_dir = torch.sign(targets)
 
    is_tie = pred_dir == 0
    correct = (pred_dir == true_dir).float()
    correct[is_tie] = 0.5  # zero prediction = no directional call = coin-flip credit

    return correct.mean().item()


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


def majority_class_baseline_preds(train_targets: torch.Tensor, n: int) -> torch.Tensor:
    """A real directional baseline: always predict the MORE COMMON direction
    seen in the training set (e.g. if the stock went up on 54% of training
    days, always predict a small positive return).
 
    This is the bar a model actually needs to clear to demonstrate genuine
    directional skill -- unlike the naive-zero baseline, this one makes a
    real directional call every time, so its accuracy reflects true base
    rate rather than a metric artifact.
 
    `train_targets` must be the TRAINING set only (never test/val), to
    avoid leaking test-period information into the baseline itself.
    `n` is the number of predictions to generate (length of the eval set).
    """
    majority_sign = torch.sign(train_targets.mean())
    if majority_sign == 0:
        majority_sign = torch.tensor(1.0)  # tie-break: default to "up"
    # Small nonzero magnitude so torch.sign() gives a clean, unambiguous
    # direction (not literally 0, which would trigger the coin-flip case).
    return torch.full((n,), majority_sign.item() * 1e-4)