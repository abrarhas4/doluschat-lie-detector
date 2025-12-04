import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


class AnswersDataset(Dataset):
    """
    Dataset: answer text + label (0/1)
    """

    def __init__(self, df: pd.DataFrame):
        self.texts = df["answer"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
        }


def compute_split_embeddings(model, tokenizer, df, device, batch_size, max_length):
    """
    For one split (train/val/test):
      - go over all answers
      - compute CLS embedding from BERT
      - return X (embeddings), y (labels)
    """
    dataset = AnswersDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Batches"):
            # DataLoader default collate:
            # batch is a dict: {"text": list[str], "label": tensor([..])}
            texts = batch["text"]
            labels = batch["label"]  # tensor

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            outputs = model(**enc, output_hidden_states=True)
            # [B, T, H]
            last_hidden = outputs.last_hidden_state
            # CLS token representation: first token
            cls_emb = last_hidden[:, 0, :]  # [B, H]

            all_embs.append(cls_emb.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    X = np.concatenate(all_embs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # CPU use kar rahe hain (safe). GPU chahiye to yahan change kar sakte ho.
    device = torch.device("cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    for split in ["train", "val", "test"]:
        path = Path(args.data_dir) / f"{split}.csv"
        df = pd.read_csv(path)
        print(f"\nProcessing {split} ({len(df)} examples)...")
        X, y = compute_split_embeddings(
            model, tokenizer, df, device, args.batch_size, args.max_length
        )
        out_path = os.path.join(args.out_dir, f"{split}_reps.npz")
        np.savez(out_path, X=X, y=y)
        print(f"Saved {split} reps to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="data/splits dir")
    parser.add_argument("--out_dir", type=str, required=True, help="where to save *_reps.npz")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    main(args)

