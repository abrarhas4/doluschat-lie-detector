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
├─ env/                    # Python virtual environment (ignored)
├─ .gitignore
└─ README.md
3. Environment Setup

The code was developed and run on an HPC cluster using Python 3.12.

3.1. On the HPC cluster (example)
module load Python/3.12.3-GCCcore-13.3.0

cd ~/projects/doluschat-lie-detector
python -m venv env/venv
source env/venv/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install numpy pandas scikit-learn transformers torch tqdm matplotlib


You can also store these dependencies in a requirements.txt and run:

pip install -r requirements.txt

4. Data Preparation

Place the DolusChat JSONL file as:

data/doluschat.jsonl


Then run:

source env/venv/bin/activate
python src/prepare_doluschat.py \
  --data_path data/doluschat.jsonl \
  --out_dir data/splits


This will create:

data/splits/train.csv

data/splits/val.csv

data/splits/test.csv

Each CSV has columns:

question

answer

label (0 = deceptive, 1 = truthful)

split (train/val/test)

5. Methods
5.1. Method 1 – TF-IDF text classifier

Idea:
Use only the answer text, convert it to TF-IDF features, and train a Logistic Regression classifier.

Training:

python src/train_text_classifier.py \
  --data_dir data/splits \
  --out_dir outputs/text_clf


This:

Fits a TF-IDF vectorizer on answer

Trains Logistic Regression to predict label

Evaluates on train/val/test

Saves the model to: outputs/text_clf/text_clf.joblib

Re-evaluation (without retraining):

python src/eval_text_classifier.py


This reloads the saved model and prints the classification report for train/val/test.

5.2. Method 2 – Classifier on BERT latent representations

Idea:
Use a frozen BERT model (bert-base-uncased) to obtain a 768-D CLS embedding for each answer, reduce to 256-D with PCA, and train Logistic Regression on these latent features.

Step 1: Extract BERT embeddings

python src/extract_hidden_states.py \
  --model_name bert-base-uncased \
  --splits train val test \
  --max_length 128


This produces:

data/reps/train_reps.npz

data/reps/val_reps.npz

data/reps/test_reps.npz

(each containing X = embeddings, y = labels).

Step 2: Train latent classifier

python src/train_latent_classifier.py \
  --reps_dir data/reps \
  --out_dir outputs/latent_clf


This:

Applies PCA to reduce embeddings (e.g. 768 → 256)

Trains Logistic Regression on the reduced features

Evaluates on train/val/test

Saves the model to: outputs/latent_clf/latent_clf.joblib

5.3. Method 3 – Reading vector / truth direction

Idea:
Find a single “truth direction” in BERT space by averaging (truthful – deceptive) embedding differences, then classify answers using the projection onto this direction.

Step 1: Compute reading vector

python src/compute_reading_vector.py \
  --model_name bert-base-uncased \
  --data_path data/doluschat.jsonl \
  --num_pairs 5000 \
  --max_length 128 \
  --out_path outputs/reading_vector/reading_vector.npy


Step 2: Evaluate using scalar scores

Assuming you already have data/reps/*.npz from Method 2:

python src/eval_reading_vector.py \
  --reps_dir data/reps \
  --reading_vector_path outputs/reading_vector/reading_vector.npy


This:

Loads the reading vector v

For each embedding h, computes s = h · v

Trains Logistic Regression on the single scalar feature s

Reports train/val/test performance

6. Results
6.1. Test accuracy comparison

On the DolusChat test split:

Method	Features	Test accuracy
1. TF-IDF + Logistic Regression	Bag-of-words TF-IDF on answer	≈ 92.5%
2. BERT latent + Logistic Regression	BERT CLS embedding (PCA to 256-D)	≈ 87.9%
3. Reading vector + Logistic Regression	1-D projection score h · v	≈ 79.2%
6.2. Discussion

Method 1 achieves the highest accuracy (~92.5%), indicating that in DolusChat,
the lexical cues (exact wording) are very strong indicators of truth vs deception.

Method 2 shows that a frozen BERT’s latent space also carries a clear
truth/deception signal, but under this simple setup (no fine-tuning, PCA + linear classifier)
it performs slightly worse than the TF-IDF baseline.

Method 3 is the most interpretable approach: by averaging (truth – lie) differences,
we obtain a single “truthfulness direction” and use the scalar projection onto this
direction as a feature. Despite its simplicity, it reaches ~79% test accuracy.

7. Plots

To generate comparison plots (e.g. test accuracy per method):

python src/plot_results.py


This will write .png plots to the plots/ directory, which can be used directly in slides or reports.

8. Acknowledgements

DolusChat dataset and original task idea come from course material / assignments on
interpretability and lie detection in large language models.

This repository contains my implementation and experiments for comparing different
representations (TF-IDF, BERT latent space, and a reading-vector direction).

