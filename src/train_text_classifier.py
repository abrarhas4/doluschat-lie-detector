import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def load_split(split_dir: str, split: str) -> pd.DataFrame:
    path = Path(split_dir) / f"{split}.csv"
    return pd.read_csv(path)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    train_df = load_split(args.data_dir, "train")
    val_df = load_split(args.data_dir, "val")
    test_df = load_split(args.data_dir, "test")

    X_train = train_df["answer"].astype(str).tolist()
    y_train = train_df["label"].astype(int).values

    X_val = val_df["answer"].astype(str).tolist()
    y_val = val_df["label"].astype(int).values

    X_test = test_df["answer"].astype(str).tolist()
    y_test = test_df["label"].astype(int).values

    clf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, 2),
                    lowercase=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=200,
                    n_jobs=args.n_jobs,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("Fitting text classifier (TF-IDF + LogisticRegression)...")
    clf.fit(X_train, y_train)

    def eval_split(X, y, name: str):
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"\n=== {name} accuracy: {acc:.4f}")
        print(classification_report(y, y_pred, digits=3))

    print("\nEvaluating on train/val/test:")
    eval_split(X_train, y_train, "Train")
    eval_split(X_val, y_val, "Val")
    eval_split(X_test, y_test, "Test")

    model_path = os.path.join(args.out_dir, "text_clf.joblib")
    joblib.dump(clf, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with train/val/test CSVs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to save the trained model and logs",
    )
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()
    main(args)

