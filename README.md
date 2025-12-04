# DolusChat Lie Detector

Detecting **truthful vs deceptive assistant answers** on the DolusChat dataset using three different methods:
1. A simple **TF-IDF text classifier**
2. A **classifier on BERT latent representations**
3. A **reading-vector–based classifier** that uses a single “truth direction” in embedding space

This repository contains all code to reproduce the experiments on an HPC cluster (and locally, with the right environment).

---

## 1. Project Overview

DolusChat is a synthetic dataset of pairs of assistant responses:
- A **truthful answer**  
- A **deceptive answer**  

for the same underlying question or situation.

The goal of this project is to **train lie detectors** that, given *only the answer text*, predict:
- `1` = **truthful**  
- `0` = **deceptive**

We implement three different approaches and compare them:

- **Method 1 – TF-IDF + Logistic Regression**  
  A classical bag-of-words model on answer text.

- **Method 2 – Classifier on BERT latent representations**  
  Uses frozen BERT embeddings (CLS token) plus a linear classifier.

- **Method 3 – Reading vector / truth direction**  
  Builds a single direction in BERT space pointing from lies to truths and classifies using a scalar score.

---

## 2. Repository Structure

```text
doluschat-lie-detector/
├─ src/                    # Python source code
│  ├─ prepare_doluschat.py         # Load JSONL, build train/val/test splits
│  ├─ train_text_classifier.py     # Method 1: TF-IDF + Logistic Regression
│  ├─ extract_hidden_states.py     # Extract BERT embeddings for each split
│  ├─ train_latent_classifier.py   # Method 2: classifier on latent reps
│  ├─ compute_reading_vector.py    # Compute truth–lie reading vector
│  ├─ eval_reading_vector.py       # Method 3: scalar-score classifier
│  ├─ plot_results.py              # Generate accuracy comparison plots
│  ├─ eval_text_classifier.py      # Re-evaluate Method 1 without retraining
│  └─ (other helpers)
│
├─ data/
│  ├─ doluschat.jsonl      # Raw DolusChat data (not committed)
│  ├─ splits/              # train.csv / val.csv / test.csv (ignored)
│  └─ reps/                # latent representations (ignored)
│
├─ outputs/                # Saved models, metrics (ignored)
├─ plots/                  # Result plots (.png) for report / slides
├─ logs/                   # Training logs (ignored)
├─ env/                    # Python virtual environm
ent (ignored)
├─ .gitignore
└─ README.md


## 3. DolusChat Lie Detector

Detecting **truthful vs deceptive assistant answers** on the DolusChat dataset using three different methods:
1. A simple **TF-IDF text classifier**
2. A **classifier on BERT latent representations**
3. A **reading-vector–based classifier** that uses a single “truth direction” in embedding space

This repository contains all code to reproduce the experiments on an HPC cluster (and locally, with the right environment).

---
