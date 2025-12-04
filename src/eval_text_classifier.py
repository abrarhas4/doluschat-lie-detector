import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def eval_split(name, answers, labels, model):
    """
    Evaluate the given model on one split (train/val/test).
    - name: string, e.g. "Train"
    - answers: Series or list of raw answer texts
    - labels: numpy array of 0/1 labels
    - model: sklearn Pipeline (TF-IDF + LogisticRegression)
    """
    # answers -> ensure string type
    X = answers.astype(str)
    y_true = labels

    # Pipeline directly does vectorization + prediction
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))


def main(args):
    # 1) Load saved Pipeline model (TF-IDF + LogisticRegression)
    print(f"Loading model from {args.model_path} ...")
    model = joblib.load(args.model_path)

    # 2) Load data splits
    print(f"Loading data from {args.data_dir} ...")
    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/val.csv")
    test_df = pd.read_csv(f"{args.data_dir}/test.csv")

    # 3) Evaluate on each split
    eval_split("Train", train_df["answer"], train_df["label"].values, model)
    eval_split("Val",   val_df["answer"],   val_df["label"].values,   model)
    eval_split("Test",  test_df["answer"],  test_df["label"].values,  model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/text_clf/text_clf.joblib",
        help="Path to saved TF-IDF + LogisticRegression pipeline model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/splits",
        help="Directory with train/val/test CSVs",
    )
    args = parser.parse_args()
    main(args)

