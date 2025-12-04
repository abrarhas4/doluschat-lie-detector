import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_doluschat(path: str) -> pd.DataFrame:
    """
    Load DolusChat from the JSONL we saved with datasets.to_json.

    Each row (line) usually has keys like:
      - "context" (dict)
      - "system_message" (str)
      - "user_query" (dict with "content")
      - "responses" (dict with "truthful", "deceptive")
      - plus some metadata (deception_difficulty, lie_type, ...)

    We just extract:
      question  = user_query["content"]
      truthful  = responses["truthful"]
      deceptive = responses["deceptive"]
    """
    path = Path(path)
    records = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # user question / query
            user_query = obj.get("user_query", {})
            if isinstance(user_query, dict):
                question = user_query.get("content", "")
            else:
                question = ""

            # responses dict
            responses = obj.get("responses", {})
            if isinstance(responses, dict):
                truthful = responses.get("truthful", "")
                deceptive = responses.get("deceptive", "")
            else:
                truthful = ""
            deceptive = "" if not isinstance(responses, dict) else responses.get("deceptive", "")

            # skip malformed
            if not truthful or not deceptive:
                continue

            records.append(
                {
                    "question": question,
                    "truthful_answer": truthful,
                    "dishonest_answer": deceptive,
                }
            )

    df = pd.DataFrame(records)
    print("Loaded DolusChat pairs shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample row:")
    print(df.head(1))
    return df


def build_answer_rows(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each row with (question, truthful_answer, dishonest_answer)
    into two rows:
      - (question, answer=truthful_answer, label=1)
      - (question, answer=dishonest_answer, label=0)
    """
    rows = []
    for _, row in df_pairs.iterrows():
        q = row["question"]
        t = row["truthful_answer"]
        d = row["dishonest_answer"]

        rows.append({"question": q, "answer": t, "label": 1})
        rows.append({"question": q, "answer": d, "label": 0})

    df = pd.DataFrame(rows)
    print("Built answer-rows shape:", df.shape)
    print("Answer-rows columns:", df.columns.tolist())
    print("Sample answer row:")
    print(df.head(1))
    return df


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DolusChat from {args.data_path} ...")
    df_pairs = load_doluschat(args.data_path)
    print("Total pairs:", len(df_pairs))

    df = build_answer_rows(df_pairs)
    print("Total answer rows:", len(df))

    # Make sure label column exists
    assert "label" in df.columns, "label column missing!"

    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # train / temp split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"],
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("Saved splits to", out_dir)
    print("  train:", len(train_df))
    print("  val  :", len(val_df))
    print("  test :", len(test_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to doluschat.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to save train/val/test CSV files",
    )
    args = parser.parse_args()
    main(args)

