import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_reps(reps_dir: str, split: str):
    data = np.load(os.path.join(reps_dir, f"{split}_reps.npz"))
    return data["X"], data["y"]


def main(args):
    # 1) Reading vector load karo
    v = np.load(args.reading_vector_path).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)

    # 2) BERT embeddings load karo (jo tum already extract kar chuke ho)
    X_train, y_train = load_reps(args.reps_dir, "train")
    X_val, y_val = load_reps(args.reps_dir, "val")
    X_test, y_test = load_reps(args.reps_dir, "test")

    print("Reps train shape:", X_train.shape)
    print("Reading vector dim:", v.shape[0])

    # 3) Har embedding ko reading vector pe project karo: s = <h, v>
    def project(X):
        return np.dot(X, v)  # [N]

    s_train = project(X_train).reshape(-1, 1)
    s_val = project(X_val).reshape(-1, 1)
    s_test = project(X_test).reshape(-1, 1)

    # 4) Sirf is 1D score pe LogisticRegression train karo
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    print("Fitting LR on scalar scores...")
    clf.fit(s_train, y_train)

    def eval_split(s, y, name: str):
        y_pred = clf.predict(s)
        acc = accuracy_score(y, y_pred)
        print(f"\n=== {name} (reading-vector) accuracy: {acc:.4f}")
        print(classification_report(y, y_pred, digits=3))

    eval_split(s_train, y_train, "Train")
    eval_split(s_val, y_val, "Val")
    eval_split(s_test, y_test, "Test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps_dir", type=str, required=True, help="dir with *_reps.npz")
    parser.add_argument(
        "--reading_vector_path",
        type=str,
        required=True,
        help="outputs/reading_vector/reading_vector.npy",
    )
    args = parser.parse_args()
    main(args)

