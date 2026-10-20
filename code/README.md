# Reproduction

## Task 1
- Noisy CE model: the first part of run3_ce_seed1__run4_elr__run5_cleaned_retrain__fusion.ipynb (plain cross entropy on noisy labels, seed 1, 60 epochs) writes predictions/t1_noisy_ce.csv from the final epoch.
- Clean reference: task1_clean_ref.ipynb (same setup, clean labels, seed 0, 60 epochs) writes predictions/t1_clean_ref.csv from the final epoch.
- Noise rate and confusion matrix: computed from data/labels.csv on the training indices.

## Task 2
The Task 1 noisy run (run3 notebook, first part) logs the four per-epoch columns to predictions/t2_dynamics.csv. The early-stopping epoch is the epoch with the highest noisy validation accuracy (epoch 37); its test predictions are saved to predictions/t2_early_stop.csv. Clean labels are used only to split the training set into correct and mislabeled subsets for logging, never for training or selection.

## Task 4
task4_elr_relabel.ipynb trains two methods, ResNet-18, 60 epochs, same optimizer and augmentation as the baseline:
- elr: cross entropy plus early-learning regularization (beta 0.7, lambda 3), seed 0. Writes predictions/t4_elr.csv.
- relabel: average the epoch-averaged softmax of the five Task 3 runs; keep a label if the ensemble agrees, replace it with the ensemble class if the ensemble disagrees with confidence at least 0.5, otherwise drop the image (weight 0). Retrain with cross entropy, seed 0. Writes predictions/t4_relabel.csv.
For each method, test predictions come from the epoch with the highest noisy validation accuracy. The best method is the one with the highest noisy validation accuracy (relabel).

## Task 5
task5_synthetic.ipynb trains on the synthetic_label column, seed 0:
- Baseline: plain cross entropy, final epoch (as in Task 1). Writes predictions/t5_baseline_synth.csv.
- Best Task 4 method (relabel): the baseline run's epoch-averaged softmax replaces the Task 3 ensemble (those models saw human labels), same keep, relabel and drop rule, then retrain. Writes predictions/t5_best_synth.csv, epoch chosen by synthetic validation accuracy only.

## Seeds
results.json lists seed 0. Individual runs: Task 3 run 1 seed 0, run 2 seed 0, run 3 seed 1, run 4 seed 4, run 5 seed 5; Task 1 clean reference, Task 4 and Task 5 seed 0.

## Task 3 (noise scores)
Notebooks, run in this order. They read data from data/ and write intermediate .npy files to outputs/:
- run1_ce_seed0.ipynb: Run 1, plain cross entropy, seed 0
- run2_sop.ipynb: Run 2, SOP
- run3_ce_seed1__run4_elr__run5_cleaned_retrain__fusion.ipynb: Run 3 (plain cross entropy, seed 1, also used for Tasks 1 and 2), Run 4 (ELR), Run 5 (cleaned retraining) and the final fusion

Five ResNet-18 runs, from scratch, 60 epochs each, noisy labels only. Each saves softmax probabilities of all 45,000 training images after every epoch.
1. Plain cross entropy, seed 0
2. SOP, seed 0
3. Plain cross entropy, seed 1
4. ELR (beta 0.7, lambda 3)
5. Cross entropy with per-image weights from the rank fusion of runs 1 to 4

Score per run: log(max other class prob) minus log(prob of given label), on the epoch-averaged probabilities. Final score is the average rank over the 5 runs, scaled to [0, 1], written to predictions/t3_noise_scores.csv.
Clean labels are used only to report precision@k and AUROC, never for training or selection.
