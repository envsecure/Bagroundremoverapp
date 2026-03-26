import torch

smooth = 1e-6  # Prevent division by zero


def dice_coef(y_pred, y_true):
    """Dice coefficient (used as a metric during training)."""
    y_true = y_true.float()
    y_pred = y_pred.float()

    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)

    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def dice_loss(y_pred, y_true):
    """Dice loss = 1 - dice_coef."""
    return 1.0 - dice_coef(y_pred, y_true)


def iou(y_pred, y_true):
    """Intersection over Union metric."""
    y_true = y_true.float()
    y_pred = y_pred.float()

    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)

    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)