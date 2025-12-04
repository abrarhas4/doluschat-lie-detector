import os

import matplotlib
matplotlib.use("Agg")  # important for HPC (no display)
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Metrics: fill from your runs
    methods = ["Text TF-IDF", "Latent BERT", "Reading vector"]

    train_acc = [0.9460, 0.8810, 0.7986]
    val_acc   = [0.9261, 0.8795, 0.7971]
    test_acc  = [0.9253, 0.8785, 0.7922]

    os.makedirs("../plots", exist_ok=True)

    # 1) Simple bar chart: Test accuracy per method
    plt.figure(figsize=(6, 4))
    x = np.arange(len(methods))
    plt.bar(x, test_acc)
    plt.xticks(x, methods, rotation=15, ha="right")
    plt.ylim(0.7, 1.0)
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy per method")
    for i, acc in enumerate(test_acc):
        plt.text(i, acc + 0.005, f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("../plots/test_accuracy_per_method.png", dpi=200)
    plt.close()

    # 2) Grouped bar chart: Train / Val / Test for each method
    plt.figure(figsize=(7, 4))
    width = 0.25
    x = np.arange(len(methods))

    plt.bar(x - width, train_acc, width, label="Train")
    plt.bar(x,         val_acc,   width, label="Val")
    plt.bar(x + width, test_acc,  width, label="Test")

    plt.xticks(x, methods, rotation=15, ha="right")
    plt.ylim(0.7, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Train / Val / Test accuracy per method")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../plots/all_splits_per_method.png", dpi=200)
    plt.close()

    print("Saved plots to ../plots/:")
    print("  - test_accuracy_per_method.png")
    print("  - all_splits_per_method.png")


if __name__ == "__main__":
    main()

