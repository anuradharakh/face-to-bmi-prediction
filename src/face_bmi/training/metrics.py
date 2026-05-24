import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def pearson_corr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) < 2:
        return 0.0

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "pearson_r": pearson_corr(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }