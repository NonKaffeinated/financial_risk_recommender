# FinSage Dataset EDA

Dataset path: `/content/drive/MyDrive/financial-news-dataset-master/Datasets`

Loaded **84,000** articles from **84** ZIP archives (48 positive ZIPs → 48,000 articles; 36 negative ZIPs → 36,000 articles).

## ZIP Inventory

Date range: `20241223` – `20260524`

| zip_file                                                | label_name   |   release_date |   articles |
|:--------------------------------------------------------|:-------------|---------------:|-----------:|
| Financial and Economic News_negative_20241223003542.zip | Negative     |       20241223 |       1000 |
| Financial and Economic News_negative_20241223072634.zip | Negative     |       20241223 |       1000 |
| Financial and Economic News_positive_20241223004000.zip | Positive     |       20241223 |       1000 |
| Financial and Economic News_positive_20241223003638.zip | Positive     |       20241223 |       1000 |
| Financial and Economic News_positive_20241224072640.zip | Positive     |       20241224 |       1000 |
| Financial and Economic News_positive_20241225072633.zip | Positive     |       20241225 |       1000 |
| Financial and Economic News_positive_20241226072651.zip | Positive     |       20241226 |       1000 |
| Financial and Economic News_negative_20241227072636.zip | Negative     |       20241227 |       1000 |
| Financial and Economic News_positive_20241228072639.zip | Positive     |       20241228 |       1000 |
| Financial and Economic News_negative_20241229072626.zip | Negative     |       20241229 |       1000 |
| Financial and Economic News_positive_20241230072647.zip | Positive     |       20241230 |       1000 |
| Financial and Economic News_negative_20241231072636.zip | Negative     |       20241231 |       1000 |
| Financial and Economic News_positive_20250105072626.zip | Positive     |       20250105 |       1000 |
| Financial and Economic News_positive_20250112072620.zip | Positive     |       20250112 |       1000 |
| Financial and Economic News_negative_20250119072618.zip | Negative     |       20250119 |       1000 |
| Financial and Economic News_negative_20250126072618.zip | Negative     |       20250126 |       1000 |
| Financial and Economic News_positive_20250202072617.zip | Positive     |       20250202 |       1000 |
| Financial and Economic News_positive_20250209072617.zip | Positive     |       20250209 |       1000 |
| Financial and Economic News_negative_20250216072617.zip | Negative     |       20250216 |       1000 |
| Financial and Economic News_positive_20250223072618.zip | Positive     |       20250223 |       1000 |
| Financial and Economic News_positive_20250302072618.zip | Positive     |       20250302 |       1000 |
| Financial and Economic News_negative_20250309072617.zip | Negative     |       20250309 |       1000 |
| Financial and Economic News_negative_20250316072616.zip | Negative     |       20250316 |       1000 |
| Financial and Economic News_positive_20250323072617.zip | Positive     |       20250323 |       1000 |
| Financial and Economic News_positive_20250330072616.zip | Positive     |       20250330 |       1000 |
| Financial and Economic News_positive_20250406072617.zip | Positive     |       20250406 |       1000 |
| Financial and Economic News_positive_20250413072616.zip | Positive     |       20250413 |       1000 |
| Financial and Economic News_positive_20250420072617.zip | Positive     |       20250420 |       1000 |
| Financial and Economic News_positive_20250427072615.zip | Positive     |       20250427 |       1000 |
| Financial and Economic News_positive_20250504072616.zip | Positive     |       20250504 |       1000 |
| Financial and Economic News_positive_20250511072615.zip | Positive     |       20250511 |       1000 |
| Financial and Economic News_negative_20250518072616.zip | Negative     |       20250518 |       1000 |
| Financial and Economic News_positive_20250525072617.zip | Positive     |       20250525 |       1000 |
| Financial and Economic News_positive_20250601072618.zip | Positive     |       20250601 |       1000 |
| Financial and Economic News_negative_20250608072616.zip | Negative     |       20250608 |       1000 |
| Financial and Economic News_positive_20250615100617.zip | Positive     |       20250615 |       1000 |
| Financial and Economic News_negative_20250622072616.zip | Negative     |       20250622 |       1000 |
| Financial and Economic News_negative_20250629072616.zip | Negative     |       20250629 |       1000 |
| Financial and Economic News_negative_20250706072617.zip | Negative     |       20250706 |       1000 |
| Financial and Economic News_negative_20250713072616.zip | Negative     |       20250713 |       1000 |
| Financial and Economic News_positive_20250720072614.zip | Positive     |       20250720 |       1000 |
| Financial and Economic News_negative_20250727072616.zip | Negative     |       20250727 |       1000 |
| Financial and Economic News_negative_20250803072616.zip | Negative     |       20250803 |       1000 |
| Financial and Economic News_positive_20250810072616.zip | Positive     |       20250810 |       1000 |
| Financial and Economic News_negative_20250817072616.zip | Negative     |       20250817 |       1000 |
| Financial and Economic News_positive_20250824072616.zip | Positive     |       20250824 |       1000 |
| Financial and Economic News_positive_20250831072616.zip | Positive     |       20250831 |       1000 |
| Financial and Economic News_negative_20250907072614.zip | Negative     |       20250907 |       1000 |
| Financial and Economic News_positive_20250914072616.zip | Positive     |       20250914 |       1000 |
| Financial and Economic News_negative_20250921072616.zip | Negative     |       20250921 |       1000 |
| Financial and Economic News_negative_20250928072617.zip | Negative     |       20250928 |       1000 |
| Financial and Economic News_positive_20251005072615.zip | Positive     |       20251005 |       1000 |
| Financial and Economic News_negative_20251012072616.zip | Negative     |       20251012 |       1000 |
| Financial and Economic News_positive_20251019072615.zip | Positive     |       20251019 |       1000 |
| Financial and Economic News_positive_20251026072616.zip | Positive     |       20251026 |       1000 |
| Financial and Economic News_positive_20251109072614.zip | Positive     |       20251109 |       1000 |
| Financial and Economic News_negative_20251116072616.zip | Negative     |       20251116 |       1000 |
| Financial and Economic News_positive_20251123072616.zip | Positive     |       20251123 |       1000 |
| Financial and Economic News_negative_20251130072616.zip | Negative     |       20251130 |       1000 |
| Financial and Economic News_positive_20251207072615.zip | Positive     |       20251207 |       1000 |
| Financial and Economic News_positive_20251214072618.zip | Positive     |       20251214 |       1000 |
| Financial and Economic News_negative_20251221072615.zip | Negative     |       20251221 |       1000 |
| Financial and Economic News_negative_20251228072614.zip | Negative     |       20251228 |       1000 |
| Financial and Economic News_positive_20260104072616.zip | Positive     |       20260104 |       1000 |
| Financial and Economic News_negative_20260111072617.zip | Negative     |       20260111 |       1000 |
| Financial and Economic News_negative_20260118072617.zip | Negative     |       20260118 |       1000 |
| Financial and Economic News_positive_20260125072616.zip | Positive     |       20260125 |       1000 |
| Financial and Economic News_positive_20260201072615.zip | Positive     |       20260201 |       1000 |
| Financial and Economic News_negative_20260208072615.zip | Negative     |       20260208 |       1000 |
| Financial and Economic News_negative_20260215072615.zip | Negative     |       20260215 |       1000 |
| Financial and Economic News_positive_20260222072616.zip | Positive     |       20260222 |       1000 |
| Financial and Economic News_negative_20260301072616.zip | Negative     |       20260301 |       1000 |
| Financial and Economic News_positive_20260308072617.zip | Positive     |       20260308 |       1000 |
| Financial and Economic News_negative_20260315072622.zip | Negative     |       20260315 |       1000 |
| Financial and Economic News_positive_20260322072615.zip | Positive     |       20260322 |       1000 |
| Financial and Economic News_positive_20260329072617.zip | Positive     |       20260329 |       1000 |
| Financial and Economic News_positive_20260405072617.zip | Positive     |       20260405 |       1000 |
| Financial and Economic News_positive_20260412072617.zip | Positive     |       20260412 |       1000 |
| Financial and Economic News_positive_20260419072616.zip | Positive     |       20260419 |       1000 |
| Financial and Economic News_negative_20260426072615.zip | Negative     |       20260426 |       1000 |
| Financial and Economic News_negative_20260503072617.zip | Negative     |       20260503 |       1000 |
| Financial and Economic News_negative_20260510072618.zip | Negative     |       20260510 |       1000 |
| Financial and Economic News_positive_20260517072617.zip | Positive     |       20260517 |       1000 |
| Financial and Economic News_positive_20260524072616.zip | Positive     |       20260524 |       1000 |

## Label Distribution

| label    |   count |
|:---------|--------:|
| Positive |   48000 |
| Negative |   36000 |

## Monthly Article Volume

|   month | label_name   |   articles |
|--------:|:-------------|-----------:|
|  202412 | Negative     |       5000 |
|  202412 | Positive     |       7000 |
|  202501 | Negative     |       2000 |
|  202501 | Positive     |       2000 |
|  202502 | Negative     |       1000 |
|  202502 | Positive     |       3000 |
|  202503 | Negative     |       2000 |
|  202503 | Positive     |       3000 |
|  202504 | Positive     |       4000 |
|  202505 | Negative     |       1000 |
|  202505 | Positive     |       3000 |
|  202506 | Negative     |       3000 |
|  202506 | Positive     |       2000 |
|  202507 | Negative     |       3000 |
|  202507 | Positive     |       1000 |
|  202508 | Negative     |       2000 |
|  202508 | Positive     |       3000 |
|  202509 | Negative     |       3000 |
|  202509 | Positive     |       1000 |
|  202510 | Negative     |       1000 |
|  202510 | Positive     |       3000 |
|  202511 | Negative     |       2000 |
|  202511 | Positive     |       2000 |
|  202512 | Negative     |       2000 |
|  202512 | Positive     |       2000 |
|  202601 | Negative     |       2000 |
|  202601 | Positive     |       2000 |
|  202602 | Negative     |       2000 |
|  202602 | Positive     |       2000 |
|  202603 | Negative     |       2000 |
|  202603 | Positive     |       3000 |
|  202604 | Negative     |       1000 |
|  202604 | Positive     |       3000 |
|  202605 | Negative     |       2000 |
|  202605 | Positive     |       2000 |

## Schema Profile (top 20 fields by presence rate)

| field                |   presence_ratio |
|:---------------------|-----------------:|
| ai_allow             |                1 |
| author               |                1 |
| categories           |                1 |
| crawled              |                1 |
| entities             |                1 |
| external_links       |                1 |
| highlightText        |                1 |
| highlightThreadTitle |                1 |
| highlightTitle       |                1 |
| language             |                1 |
| ord_in_thread        |                1 |
| published            |                1 |
| rating               |                1 |
| sentiment            |                1 |
| syndication          |                1 |
| text                 |                1 |
| thread               |                1 |
| title                |                1 |
| updated              |                1 |
| url                  |                1 |

## Data Quality

| metric          |    value |
|:----------------|---------:|
| has_title_rate  |    0.998 |
| has_text_rate   |    1     |
| avg_title_len   |   78.35  |
| avg_text_len    | 3371.14  |
| median_text_len | 2572     |

## Text Length Percentiles

| field     |   mean |   p10 |   p25 |   p50 |   p75 |    p90 |    p95 |   p99 |
|:----------|-------:|------:|------:|------:|------:|-------:|-------:|------:|
| title_len |   78.4 |    50 |    61 |    75 |    92 |  109   |  123   |   164 |
| text_len  | 3371.1 |   352 |  1169 |  2572 |  4317 | 6617.1 | 8380.1 | 17786 |

## Ticker Keyword Coverage

| ticker   | keyword   |   Positive |   Negative |   total |
|:---------|:----------|-----------:|-----------:|--------:|
| AAPL     | apple     |       1147 |        993 |    2140 |
| AMZN     | amazon    |        897 |        604 |    1501 |
| GOOGL    | google    |       1295 |        977 |    2272 |
| JPM      | jpmorgan  |        946 |        711 |    1657 |
| META     | meta      |       3602 |       2402 |    6004 |
| META     | facebook  |       2026 |       1460 |    3486 |
| MSFT     | microsoft |        927 |        494 |    1421 |
| NFLX     | netflix   |        370 |        125 |     495 |
| NVDA     | nvidia    |       1939 |        954 |    2893 |
| TSLA     | tesla     |       1042 |        585 |    1627 |

## Financial Feature Template

The financial anomaly detector reads six Yahoo Finance ratios live at inference time. This template records the expected feature schema for the ten baseline tickers.

| ticker   |   debtToEquity |   profitMargins |   revenueGrowth |   currentRatio |   returnOnEquity |   operatingMargins |
|:---------|---------------:|----------------:|----------------:|---------------:|-----------------:|-------------------:|
| AAPL     |         79.548 |         0.27152 |           0.166 |          1.07  |          1.41471 |            0.32275 |
| MSFT     |         30.271 |         0.39342 |           0.183 |          1.283 |          0.34014 |            0.46326 |
| JNJ      |         67.73  |         0.21834 |           0.099 |          1.025 |          0.26416 |            0.27408 |
| JPM      |        nan     |         0.33936 |           0.127 |        nan     |          0.16465 |            0.43741 |
| PG       |         67.651 |         0.19162 |           0.074 |          0.732 |          0.31112 |            0.23047 |
| V        |         67.233 |         0.51679 |           0.171 |          1.088 |          0.60349 |            0.67346 |
| UNH      |         73.982 |         0.02678 |           0.02  |          0.798 |          0.12176 |            0.08047 |
| HD       |        455.218 |         0.08411 |           0.048 |          1.045 |          1.2838  |            0.11926 |
| MA       |        282.059 |         0.45876 |           0.158 |          0.981 |          2.32076 |            0.60836 |
| DIS      |         41.069 |         0.1154  |           0.065 |          0.679 |          0.1101  |            0.15512 |
