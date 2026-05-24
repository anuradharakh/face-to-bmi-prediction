import sys

sys.path.append("src")

from face_bmi.training.metrics import regression_metrics


def main():
    y_true = [20, 25, 30, 35, 40]
    y_pred = [21, 24, 31, 34, 39]

    metrics = regression_metrics(y_true, y_pred)

    print("Metrics test passed.")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()