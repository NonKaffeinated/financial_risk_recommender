# FinSage Benchmark and Evaluation

Dataset path: `/content/drive/MyDrive/financial-news-dataset-master/Datasets`

Loaded **84,000** articles (48,000 positive, 36,000 negative, imbalance ratio 1.33:1). Train/test split: 80/20 stratified (random_state=42).

## Sentiment Model Benchmark (single split)

`tfidf_logreg_production` replicates the 500-char truncation and hyperparameters in `sentiment.py`. `tfidf_logreg_balanced` and `tfidf_rf_balanced` use full article text with `class_weight='balanced'`. VADER runs zero-shot on the test set (compound threshold ≥ 0.05).

| model                   |   accuracy |   precision_binary |   recall_binary |   f1_binary |   precision_weighted |   recall_weighted |   f1_weighted |   roc_auc |
|:------------------------|-----------:|-------------------:|----------------:|------------:|---------------------:|------------------:|--------------:|----------:|
| vader_baseline          |     0.6805 |             0.663  |          0.8968 |      0.7624 |               0.6961 |            0.6805 |        0.6554 |  nan      |
| tfidf_logreg_production |     0.8455 |             0.8543 |          0.8798 |      0.8668 |               0.8452 |            0.8455 |        0.8451 |    0.9194 |
| tfidf_logreg_balanced   |     0.8294 |             0.868  |          0.8273 |      0.8471 |               0.8317 |            0.8294 |        0.8299 |    0.9085 |
| tfidf_rf_balanced       |     0.8527 |             0.8562 |          0.8921 |      0.8738 |               0.8525 |            0.8527 |        0.8521 |    0.933  |

## Confusion Matrices

| model                   |   true_negative |   false_positive |   false_negative |   true_positive |
|:------------------------|----------------:|-----------------:|-----------------:|----------------:|
| vader_baseline          |            2824 |             4376 |              991 |            8609 |
| tfidf_logreg_production |            5759 |             1441 |             1154 |            8446 |
| tfidf_logreg_balanced   |            5992 |             1208 |             1658 |            7942 |
| tfidf_rf_balanced       |            5762 |             1438 |             1036 |            8564 |

## LogReg 5-Fold Cross-Validation

Vectorizer is fitted inside each fold via `sklearn.Pipeline` to prevent data leakage.

| model                      |   cv_folds |   mean_accuracy |   std_accuracy |   mean_f1_weighted |   std_f1_weighted |   mean_roc_auc |   std_roc_auc |
|:---------------------------|-----------:|----------------:|---------------:|-------------------:|------------------:|---------------:|--------------:|
| tfidf_logreg_production_cv |          5 |          0.8454 |         0.0031 |             0.8449 |            0.0031 |         0.92   |        0.0022 |
| tfidf_logreg_balanced_cv   |          5 |          0.8282 |         0.005  |             0.8288 |            0.005  |         0.9073 |        0.0027 |

## FinBERT Benchmark

Evaluated on a 10,000-article stratified sample. Label mapping: FinBERT `positive` → 1, all others → 0.

| model            |   accuracy |   precision_binary |   recall_binary |   f1_binary |   precision_weighted |   recall_weighted |   f1_weighted |   roc_auc |   articles_sampled |   inference_seconds |   sec_per_article |
|:-----------------|-----------:|-------------------:|----------------:|------------:|---------------------:|------------------:|--------------:|----------:|-------------------:|--------------------:|------------------:|
| finbert_prosusai |     0.7383 |             0.8532 |          0.5756 |      0.6874 |               0.7665 |            0.7383 |        0.7312 |       nan |              10000 |                5468 |             0.547 |

## Financial Anomaly Detection Benchmark

Isolation Forest evaluated on 16 labeled tickers (anomaly_score threshold = 0.5). Healthy tickers held out from the training set.

| ticker   |   true_label |   anomaly_score |   flag_count |   predicted_label |   correct |
|:---------|-------------:|----------------:|-------------:|------------------:|----------:|
| NVDA     |            0 |            0.29 |            0 |                 0 |         1 |
| GOOGL    |            0 |            0.28 |            0 |                 0 |         1 |
| AMZN     |            0 |            0.27 |            0 |                 0 |         1 |
| META     |            0 |            0.27 |            0 |                 0 |         1 |
| COST     |            0 |            0.28 |            0 |                 0 |         1 |
| TSLA     |            0 |            0.29 |            0 |                 0 |         1 |
| ADBE     |            0 |            0.32 |            1 |                 0 |         1 |
| CRM      |            0 |            0.33 |            1 |                 0 |         1 |
| BYND     |            1 |            0.43 |            2 |                 0 |         0 |
| AMC      |            1 |            0.5  |            3 |                 1 |         1 |
| NKLA     |            1 |            0.38 |            1 |                 0 |         0 |
| LCID     |            1 |            0.48 |            3 |                 0 |         0 |
| BLNK     |            1 |            0.49 |            3 |                 0 |         0 |
| BBBY     |            1 |            0.38 |            1 |                 0 |         0 |
| RIDE     |            1 |            0.38 |            1 |                 0 |         0 |
| SPCE     |            1 |            0.49 |            3 |                 0 |         0 |

**Accuracy:** 0.562 | **Precision (risky):** 1.0 | **Recall (risky):** 0.125 | **F1 (risky):** 0.222

## Composite Risk Model Benchmark

Both methods evaluated against labeled tickers using neutral sentiment to isolate the financial aggregation logic.

### Per-ticker results

| ticker   | expected_level   | rule_based_level   |   rule_based_trust |   rule_based_correct | content_based_level   |   content_based_trust |   content_based_correct |
|:---------|:-----------------|:-------------------|-------------------:|---------------------:|:----------------------|----------------------:|------------------------:|
| NVDA     | Low              | Low                |               67.9 |                    1 | Medium                |                  59.1 |                       0 |
| GOOGL    | Low              | Low                |               68.7 |                    1 | Medium                |                  59.4 |                       0 |
| AMZN     | Low              | Low                |               69.6 |                    1 | Medium                |                  59.8 |                       0 |
| META     | Low              | Low                |               69.6 |                    1 | Medium                |                  59.8 |                       0 |
| COST     | Low              | Low                |               68.7 |                    1 | Medium                |                  59.4 |                       0 |
| BYND     | High             | Medium             |               56   |                    0 | Medium                |                  48.3 |                       0 |
| AMC      | High             | Medium             |               50   |                    0 | Medium                |                  44.6 |                       0 |
| NKLA     | High             | Medium             |               60.2 |                    0 | Medium                |                  52.4 |                       0 |
| LCID     | High             | Medium             |               51.7 |                    0 | Medium                |                  44.9 |                       0 |
| BLNK     | High             | Medium             |               50.8 |                    0 | Medium                |                  44.8 |                       0 |

### Method summary

| method        |   accuracy |   spearman_r |   mean_trust_healthy |   mean_trust_risky |
|:--------------|-----------:|-------------:|---------------------:|-------------------:|
| rule_based    |        0.5 |        0.876 |                 68.9 |               53.7 |
| content_based |        0   |        0.876 |                 59.5 |               47   |

## Remaining Evaluation Gaps

| module                      | implemented                                                                        | remaining_limitation                                                                                                                                                                                                             |
|:----------------------------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| financial.py                | 10-ticker labeled benchmark with anomaly_score threshold at 0.5                    | Small sample (10 tickers) limits statistical power; yfinance returns current live data rather than point-in-time historical snapshots, so results for recovering firms may not reflect their distressed state                    |
| risk.py                     | Per-ticker rule_based vs content_based comparison with accuracy and Spearman r     | Neutral sentiment proxy isolates financial signals but does not test the full sentiment-weighted blend; content_based method relies on only 6 hardcoded reference profiles, which limits its generalization to new company types |
| sentiment.py (FinBERT path) | Side-by-side F1 and inference time comparison on a 2,000-article stratified sample | 2,000-article sample may underestimate variance; full-dataset FinBERT inference would require GPU acceleration to be practical; FinBERT neutral-to-negative mapping may suppress recall for ambiguous articles                   |
| benchmark_models.py         | 5-fold stratified CV for both LogReg variants via sklearn Pipeline                 | Random Forest evaluated on a single split only (5-fold CV with 300 trees on 83K documents would take 30+ minutes); no temporal split to test model drift on newer articles                                                       |
