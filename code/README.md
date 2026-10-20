# Reproduction

## Setup (all tasks)
1. Copy the provided `data/` folder and `model.py` into a Google Drive folder named `CV Bootcamp` (path `/content/drive/MyDrive/CV Bootcamp`).
2. For each notebook below: open it in Google Colab, choose Runtime > Change runtime type > GPU, then Runtime > Run all.
3. Notebooks write prediction files to `CV Bootcamp/outputs/predictions/` and intermediate `.npy` files to `CV Bootcamp/outputs/`. Copy the prediction files into `predictions/`.

To run outside Colab: in the first cell of each notebook, delete the two `google.colab` / `drive.mount` lines and set `DATA` to the folder containing `data/` and `model.py`.

Training setup for every run: ResNet-18 from `model.py`, from scratch, SGD (momentum 0.9, Nesterov, weight decay 5e-4), learning rate 0.1 with cosine decay, batch size 128, 60 epochs, random crop (pad 4) and horizontal flip, mixed precision. Each run takes about 35 to 40 minutes on one GPU.

## Task 1: noise rate, confusion matrix, baselines

Noise rate and 10x10 confusion matrix (no GPU). Run from the folder that contains `data/`:

```
python -c "import numpy as np, pandas as pd; lab = pd.read_csv('data/labels.csv').set_index('index'); t = lab.loc[np.loadtxt('data/train_idx.txt', dtype=int)]; print('noise rate %:', round(100 * (t.noisy_label != t.clean_label).mean(), 2)); cm = pd.crosstab(t.clean_label, t.noisy_label, rownames=['clean'], colnames=['noisy']); print(cm)"
```
Expected: noise rate 40.24. Rows are clean labels, columns noisy labels, class k is line k+1 of `data/classes.txt`.

| Output | Command | Expected |
|---|---|---|
| `t1_noisy_ce.csv` | Run cells 1 to 5 of `run3_ce_seed1__run4_elr__run5_cleaned_retrain__fusion.ipynb` (seed 1, final epoch). The later cells of this notebook need the Task 3 files, see Task 3 | val acc (noisy labels) 0.4938 |
| `t1_clean_ref.csv` | Run all in `task1_clean_ref.ipynb` (clean labels, seed 0, final epoch) | val acc (noisy labels) 0.5918 |

## Task 2: training dynamics and early stopping

Produced by the same run as `t1_noisy_ce.csv` (cells 1 to 5 of `run3_ce_seed1__run4_elr__run5_cleaned_retrain__fusion.ipynb`):

| Output | Contents |
|---|---|
| `t2_dynamics.csv` | Per epoch: train acc on correctly labeled and mislabeled subsets, val acc with noisy and clean labels |
| `t2_early_stop.csv` | Test predictions from the epoch with the highest noisy validation accuracy (epoch 37, 0.5512) |

Clean labels are used only to split the training set into correct and mislabeled subsets for logging.

## Task 3: noise scores

Run all, in this order (each later step reads the `.npy` files saved by earlier ones):

| Order | Notebook | Run |
|---|---|---|
| 1 | `run1_ce_seed0.ipynb` | Run 1: plain cross entropy, seed 0 |
| 2 | `run2_sop.ipynb` | Run 2: SOP, seed 0 |
| 3 | `run3_ce_seed1__run4_elr__run5_cleaned_retrain__fusion.ipynb` | Run 3: plain cross entropy, seed 1 (cells 1 to 5); Run 4: ELR, seed 4 (cells 7 to 8); Run 5: cross entropy with per-image weights from the rank fusion of runs 1 to 4, seed 5 (cells 9 to 11); final fusion (cell 12) |

Each run saves the softmax probabilities of all 45,000 training images after every epoch. Score per run: log(max other class prob) minus log(prob of given label), on the epoch-averaged probabilities. The final score is the average rank over the 5 runs, scaled to [0, 1], written to `outputs/predictions/t3_noise_scores.csv` by the last cell. (`run1_ce_seed0.ipynb` also writes an earlier single-run score file to `outputs/`; it is not the submitted one.)

Clean labels are used only to report precision@k (0.880, k = 18,109) and AUROC (0.965), never for training or selection.

## Task 4: robust training

Run all in `task4_elr_relabel.ipynb` (needs the Task 3 `.npy` files in `outputs/`):

| Output | Method | Expected val acc (noisy labels) |
|---|---|---|
| `t4_elr.csv` | Cross entropy plus early-learning regularization (beta 0.7, lambda 3), seed 0 | 0.5794 |
| `t4_relabel.csv` | Average the five Task 3 runs; keep a label if they agree, replace it with their class if they disagree with confidence at least 0.5, otherwise drop the image (weight 0); retrain with cross entropy, seed 0 | 0.5806 (best) |

For each method, test predictions come from the epoch with the highest noisy validation accuracy. The best method is chosen the same way.

## Task 5: real vs synthetic noise

Run all in `task5_synthetic.ipynb` (trains on the `synthetic_label` column, seed 0):

| Output | Method | Expected val acc (synthetic labels) |
|---|---|---|
| `t5_baseline_synth.csv` | Plain cross entropy, final epoch (as in Task 1) | 0.4268 |
| `t5_best_synth.csv` | Relabel (best Task 4 method). The baseline run's epoch-averaged predictions replace the Task 3 ensemble, since those models saw human labels; same keep, relabel and drop rule, epoch chosen by synthetic validation accuracy only | 0.5254 |

## Seeds
`results.json` lists seed 0. Individual runs: Task 3 runs 1 to 5 use seeds 0, 0, 1, 4, 5 (run 3 is also the Task 1 noisy model and the Task 2 run); Task 1 clean reference, Task 4 and Task 5 use seed 0. GPU nondeterminism means reruns can differ slightly in the last digits.

## Format check
From the folder that contains `data/`:
```
python check_format.py <submission folder or zip>
```
