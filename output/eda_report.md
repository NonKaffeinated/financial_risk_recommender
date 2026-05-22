# FinSage Dataset EDA

Dataset path: `financial-news-dataset/Datasets/extracted`

Loaded 83000 news records from the Webhose financial-news-dataset.

## Label distribution

| label    |   count |
|:---------|--------:|
| Positive |   47000 |
| Negative |   36000 |

## Schema profile

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

## Data quality

| metric          |    value |
|:----------------|---------:|
| has_title_rate  |    0.998 |
| has_text_rate   |    1     |
| avg_title_len   |   78.36  |
| avg_text_len    | 3368.29  |
| median_text_len | 2571     |

## Ticker coverage

| ticker   |   matches |
|:---------|----------:|
| AAPL     |      2123 |
| AMZN     |      1481 |
| GOOGL    |      2235 |
| JPM      |      1636 |
| META     |      9414 |
| MSFT     |      1402 |
| NFLX     |       495 |
| NVDA     |      2743 |
| TSLA     |      1615 |

## Financial feature template

The financial model uses live Yahoo Finance fields instead of a committed training file, so this EDA exports the expected six-feature template for baseline tickers.

| ticker   |   debtToEquity |   profitMargins |   revenueGrowth |   currentRatio |   returnOnEquity |   operatingMargins |
|:---------|---------------:|----------------:|----------------:|---------------:|-----------------:|-------------------:|
| AAPL     |            nan |             nan |             nan |            nan |              nan |                nan |
| MSFT     |            nan |             nan |             nan |            nan |              nan |                nan |
| JNJ      |            nan |             nan |             nan |            nan |              nan |                nan |
| JPM      |            nan |             nan |             nan |            nan |              nan |                nan |
| PG       |            nan |             nan |             nan |            nan |              nan |                nan |
| V        |            nan |             nan |             nan |            nan |              nan |                nan |
| UNH      |            nan |             nan |             nan |            nan |              nan |                nan |
| HD       |            nan |             nan |             nan |            nan |              nan |                nan |
| MA       |            nan |             nan |             nan |            nan |              nan |                nan |
| DIS      |            nan |             nan |             nan |            nan |              nan |                nan |
