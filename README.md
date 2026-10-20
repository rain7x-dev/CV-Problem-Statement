# Learning from Messy Human Labels (CIFAR-10N)

Inter-IIT Tech Meet 15.0 CV Bootcamp. Aryan Prasad (25B2493).

ResNet-18 (provided `model.py`) trained from scratch on CIFAR-10 with human label noise (CIFAR-10N, 40.24% of training labels wrong). Budget: at most 60 epochs per network. Clean labels are used only for measuring, never for training or model selection.

## Results

| Task | What | Result |
|---|---|---|
| 1 | Noise rate (45,000 training images) | 40.24% |
| 1 | Noisy-label CE model, final epoch, val acc (noisy labels) | 0.4938 |
| 1 | Clean-label reference, final epoch, val acc (noisy labels) | 0.5918 |
| 2 | Memorization start / early-stop epoch | 37 / 37 |
| 2 | Val acc (noisy labels) at early stop | 0.5512 |
| 3 | Mislabel detection, rank fusion of 5 runs | precision@k 0.880, AUROC 0.965 |
| 4 | ELR, val acc (noisy labels) | 0.5794 |
| 4 | Relabel with Task 3 ensemble, val acc (noisy labels), best | 0.5806 |
| 5 | CE baseline on synthetic labels, val acc (synthetic labels) | 0.4268 |
| 5 | Relabel on synthetic labels, val acc (synthetic labels) | 0.5254 |

Details and answers to the report questions: `report.md`. Exact numbers: `results.json`.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Notebooks for every run, and `code/README.md` mapping each notebook to its output files |
| `predictions/` | All submitted prediction files (`t1_*` to `t5_*`) |
| `results.json` | Filled results template |
| `report.md` | Answers to Q1 to Q5 |
| `model.py` | Provided ResNet-18 |
| `check_format.py` | Provided format checker |

## How to run

1. Put the provided `data/` folder (images, `labels.csv`, index files) next to `model.py`. It is not included here because of its size.
2. The notebooks were run on Google Colab with the project in Google Drive at `/content/drive/MyDrive/CV Bootcamp`. Elsewhere, remove the two Drive lines in the first cell and set `DATA` to the folder containing `data/` and `model.py`.
3. Run the notebooks in the order given in `code/README.md`. Each training run takes about 35 to 40 minutes on a single T4 or RTX 4060 GPU.
4. Check the output format: `python check_format.py <this folder>` (needs `data/` present).

Training setup for every run: SGD (momentum 0.9, Nesterov, weight decay 5e-4), learning rate 0.1 with cosine decay, batch size 128, 60 epochs, random crop (pad 4) and horizontal flip, mixed precision.
