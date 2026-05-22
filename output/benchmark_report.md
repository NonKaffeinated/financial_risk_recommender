# FinSage Benchmark and Evaluation

Dataset path: `financial-news-dataset/Datasets/extracted`

Loaded 83000 records for sentiment benchmarking.

## Sentiment benchmark

| model                         |   accuracy |   precision_weighted |   recall_weighted |   f1_weighted |   roc_auc_ovr_weighted |
|:------------------------------|-----------:|---------------------:|------------------:|--------------:|-----------------------:|
| tfidf_logreg_baseline         |     0.8396 |               0.8393 |            0.8396 |        0.8391 |                    nan |
| tfidf_random_forest_benchmark |     0.8648 |               0.8646 |            0.8648 |        0.8645 |                    nan |

## Models still needing evaluation work

| module                     | current_state                                                         | still_needed                                                                                                                     | recommended_metrics                                                    |
|:---------------------------|:----------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|
| financial.py               | Isolation Forest on six Yahoo Finance ratios and ten baseline tickers | Offline labeled benchmark dataset, holdout design, and comparison against Logistic Regression, Random Forest, and XGBoost        | ROC-AUC, PR-AUC, recall for risky firms, calibration, confusion matrix |
| risk.py                    | Rule-based weighted blend plus prototype content-based method         | Frozen benchmark table mapping company snapshots to trust or risk labels and a comparison of rule-based vs content-based scoring | Risk-band accuracy, MAE on trust score, Spearman ranking correlation   |
| sentiment.py advanced path | FinBERT stub exists but is not benchmarked                            | Side-by-side evaluation against TF-IDF baseline on the same splits and ticker-group split                                        | Weighted F1, per-class recall, confusion matrix, inference time        |

## Notes


