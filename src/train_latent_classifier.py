import argparse
import os

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_reps(reps_dir: str, split: str):
    data = np.load(os.path.join(reps_dir, f"{split}_reps.npz"))
    return data["X"], data["y"]


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train = load_reps(args.reps_dir, "train")
    X_val, y_val = load_reps(args.reps_dir, "val")
    X_test, y_test = load_reps(args.reps_dir, "test")

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    pca = None
    if args.pca_dim > 0 and args.pca_dim < X_train.shape[1]:
        print(f"Running PCA to {args.pca_dim} dimensions...")
        pca = PCA(n_components=args.pca_dim, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        print("After PCA dim:", X_train.shape[1])
    else:
        print("PCA disabled or dims not valid, using raw embeddings.")

    clf = LogisticRegression(
        max_iter=200,
        n_jobs=args.n_jobs,
        class_weight="balanced",
    )

    print("Fitting latent classifier...")
    clf.fit(X_train, y_train)

    def eval_split(X, y, name: str):
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"\n=== {name} accuracy: {acc:.4f}")
        print(classification_report(y, y_pred, digits=3))

    eval_split(X_train, y_train, "Train")
    eval_split(X_val, y_val, "Val")
    eval_split(X_test, y_test, "Test")

    joblib.dump(
        {"classifier": clf, "pca": pca},
        os.path.join(args.out_dir, "latent_clf.joblib"),
    )
    print(f"Saved latent classifier to {os.path.join(args.out_dir, 'latent_clf.joblib')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps_dir", type=str, required=True, help="dir with *_reps.npz")
    parser.add_argument("--out_dir", type=str, required=True, help="where to save model")
    parser.add_argument("--pca_dim", type=int, default=256, help="0 to disable PCA")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()
    main(args)

