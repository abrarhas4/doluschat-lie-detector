import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


class PairDataset(Dataset):
    """
    Dataset of (truth, lie) answer text pairs.
    """

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def load_pairs(path: str, max_pairs: int | None = None):
    """
    Read doluschat.jsonl and extract (truthful, deceptive) answer pairs
    from obj["responses"]["truthful"] and obj["responses"]["deceptive"].
    """
    path = Path(path)
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            responses = obj.get("responses", {})
            if not isinstance(responses, dict):
                continue

            truthful = responses.get("truthful", "")
            deceptive = responses.get("deceptive", "")

            if not truthful or not deceptive:
                continue

            records.append({"truth": truthful, "lie": deceptive})
            if max_pairs is not None and len(records) >= max_pairs:
                break

    print("Loaded pairs for reading vector:", len(records))
    return records


def compute_reading_vector(model, tokenizer, records, device, batch_size, max_length):
    """
    For each pair (truth, lie), compute CLS embeddings and take mean of (truth - lie).
    """
    dataset = PairDataset(records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    diffs = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Pairs batches"):
            truths = batch["truth"]
            lies = batch["lie"]

            enc_t = tokenizer(
                truths,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc_l = tokenizer(
                lies,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            enc_t = {k: v.to(device) for k, v in enc_t.items()}
            enc_l = {k: v.to(device) for k, v in enc_l.items()}

            out_t = model(**enc_t, output_hidden_states=True)
            out_l = model(**enc_l, output_hidden_states=True)

            cls_t = out_t.last_hidden_state[:, 0, :]  # [B, H]
            cls_l = out_l.last_hidden_state[:, 0, :]  # [B, H]

            d = cls_t - cls_l  # [B, H]
            diffs.append(d.cpu().numpy())

    D = np.concatenate(diffs, axis=0)  # [N, H]
    v = D.mean(axis=0)                 # [H]
    # normalize
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def main(args):
    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    records = load_pairs(args.data_path, max_pairs=args.max_pairs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    v = compute_reading_vector(
        model,
        tokenizer,
        records,
        device,
        args.batch_size,
        args.max_length,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "reading_vector.npy")
    np.save(out_path, v)
    print("Saved reading vector to", out_path)
    print("Vector shape:", v.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="doluschat.jsonl")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_pairs", type=int, default=5000, help="subsample for speed")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)

