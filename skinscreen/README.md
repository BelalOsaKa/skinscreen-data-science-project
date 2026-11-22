# SkinScreen — AID427 Project Phase 1

Multimodal skin-lesion classification on **PAD-UFES-20** (smartphone images + patient clinical data),
6 classes: ACK, BCC, MEL, NEV, SCC, SEK (plus a cancer vs non-cancer referral view).

## Quick start

```bash
pip install -r requirements.txt
# 1. download PAD-UFES-20 and extract it into data/pad-ufes-20/  (see data/README.md)
# 2. run everything (EDA -> reduction -> clustering -> selection -> models -> report)
python run_pipeline.py            # full run
python run_pipeline.py --fast     # quick check (smaller searches)
```

Options: `--data-dir PATH`, `--output-dir PATH`, `--smote` (SMOTE inside training folds),
`--cnn` (add ResNet-50 embeddings, needs torch/torchvision), `--skip eda reduction clustering selection`.

Results: `outputs/figures/*.png`, `outputs/tables/*.csv`, and a summary with interpretation in
`outputs/REPORT.md`. Image features are cached in `outputs/cache/`, so later runs skip extraction.

## How the code covers the Phase 1 cover sheet

| Cover-sheet item | Where | Main outputs |
|---|---|---|
| Problem introduction / dataset description | this README, `src/config.py`, `src/data_loading.py` | `eda_overview.csv` |
| Data analysis & preprocessing | `src/eda.py`, `src/preprocessing.py`, `src/image_features.py` | `eda_01…09_*.png` |
| Visualisation before/after reduction & clustering | `src/reduction.py`, `src/clustering.py` | `red_03/04_*.png`, `clu_02_*.png` |
| Linear reduction: PCA, LDA, SVD | `reduction.linear_reduction` | `red_01…03`, `red_linear_summary.csv` |
| Non-linear: Kernel PCA, SNE, t-SNE, UMAP, metric & non-metric MDS, Isomap, LLE | `reduction.nonlinear_reduction` (SNE is implemented from scratch) | `red_04_*`, `red_06_sne_convergence.png`, `red_nonlinear_quality.csv` |
| Reduction as preprocessing | `reduction.downstream_comparison` | `red_05_*`, `red_downstream_comparison.csv` |
| Clustering before vs after reduction | `clustering.run_clustering` (KMeans, Ward, GMM; silhouette, DB, CH, ARI, NMI, purity) | `clu_01…04`, `clu_metrics_before_after.csv` |
| Feature selection: filter | `feature_selection.filter_methods` (variance, correlation, ANOVA-F, MI, χ²) | `fs_01_*` |
| Feature selection: wrapper | `feature_selection.wrapper_methods` (SFS forward, SFS backward, RFE, RFECV) | `fs_02_rfecv.png` |
| Feature selection: embedded | `feature_selection.embedded_methods` (L1, L2, Elastic Net, RF importance) | `fs_03_embedded.png` |
| Feature selection: GA & metaheuristics | `feature_selection.genetic_algorithm`, `binary_pso`, `simulated_annealing` (from scratch) | `fs_04_*_convergence.png`, `fs_comparison.csv` |
| Model implementations | `src/models.py` (LogReg, LDA, NB, KNN, DT, RF, ExtraTrees, HistGB, SVM, MLP, XGBoost) | |
| 80/20 split + all metrics | `evaluation.evaluate_models` (patient-level split) | `eval_holdout_all_metrics.csv` |
| k-fold CV + average accuracy | same (5-fold, patient-grouped) | `eval_kfold_cv.csv`, `eval_05_cv_folds.png` |
| Confusion matrix, accuracy, error rate, precision, recall, F1, ROC-AUC | `evaluation.compute_metrics`, `plot_confusion`, `plot_roc` | `eval_01…03` |
| Over/underfitting interpretation & comparisons | `evaluation.diagnose_fit`, learning curve, modality comparison | `eval_04`, `eval_06`, `eval_07` |
| Comparison with related work (table) | `src/report.py` | `related_work_comparison.csv` |

## Design decisions (worth mentioning in the report)

- **No leakage:** the `biopsed` column and the IDs are dropped. The 80/20 split and every CV are
  **patient-level** (`StratifiedGroupKFold`), because one patient can have several photos.
  Imputers, scalers and SMOTE are fitted inside each training fold. Feature selection uses
  only the training set, and the final subset is chosen by inner-CV score, never by test score.
- **Missing data:** `UNK` and blank answers count as missing. Numeric values are median-imputed,
  categorical ones get their own `MISSING` level, binary ones use the mode, and every incomplete
  column gets an explicit `<col>_missing` indicator.
- **Noise:** numeric outliers are winsorised (3×IQR). Images are colour-normalised
  (Shades-of-Gray), median-filtered and resized to 128×128. Image-quality statistics (resolution,
  brightness, sharpness) appear in the EDA only and are **not** model features, so the model
  cannot learn which phone took the photo.
- **Imbalance (16:1):** class weights (or balanced sample weights for XGBoost), optional SMOTE,
  and balanced accuracy / macro-F1 as the main metrics.
- **Image features:** colour moments in RGB/HSV/Lab, HSV histograms, GLCM texture, LBP, and
  Otsu-segmented lesion shape and colour contrast (76 features). `--cnn` adds ResNet-50 embeddings.

## Before submitting

- The literature numbers in `src/report.py` (`RELATED_WORK`) come from search-result abstracts.
  **Check each one against the original paper.** The Pacheco & Krohling (2020) baseline row needs
  its numbers filled in.
- Runtime on the real 2,298 images has not been measured yet. My estimate is tens of minutes on a
  laptop; feature extraction, backward SFS and non-metric MDS are the slowest steps. Use `--fast`
  to iterate.
