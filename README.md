# BTC limit-order-book benchmark 

Five published LOB models and one custom model are trained and evaluated on identical samples from the Kaggle BTCUSDT perpetual file (Binance, Jan 9–20 2023, 10 levels, 250 ms snapshots).

## Setup

`uv sync`, then download `1-09-1-20.csv` from the Kaggle dataset (link under Sources) into `data/raw/`; its SHA-256 is in `data/raw/data_manifest.json`.

## Files

- `data.py`: processed-book loading, samples, labels, input representations
- `models.py`: the six models and their settings
- `training.py`: `train`, `fit`, `test`
- `benchmark.ipynb`: CSV parsing, data report, one cell per model, results

## Protocol

- **Splits:** train Jan 9–17, validation Jan 18, test Jan 19–20 (TLOB's split of this file).
- **Target:** `(mean(mid[t+1..t+k]) − mid[t]) / mid[t]` at snapshots where the mid-price just moved.
- **Classes:** up, stationary or down, using θ = the 1/3 quantile of training |returns| per horizon. At 0.25 s and 0.5 s over a third of returns are exactly zero, so θ = 0 and the classes are down / mid unchanged / up (23/54/23% and 30/41/29% of training samples).
- **Horizons:** 1 / 2 / 4 / 8 / 20 snapshots (0.25, 0.5, 1, 2, 5 s). All normalisers and thresholds are fitted on training data only.
- **Metrics:** macro-F1 and Pearson/Spearman of the model's score against the realised return. For return forecasts also MAE, RMSE and R²_OS against the test-set mean (Kolm eq. 18, Siamese eq. 9), pooled over both test days, where Kolm averages per day. All return metrics use the raw, uncapped returns.

## Models and settings

| Model | Lookback | Input | Optimiser | Batch | Stopping | Max epochs | Loss |
|---|---|---|---|---|---|---|---|
| deeplob | 100 | z-scored book | Adam 1e-2, ε = 1 | 32 | val accuracy, patience 20 | 100 (compute cap) | CE |
| multihorizon_deeplob | 50 | z-scored book | Adam 1e-3 | 32 | none, keep best val loss | 50 | CE per horizon |
| tlob | 128 | z-scored book → BiN | Adam 1e-4, EMA 0.999 | 128 | val loss, patience 1, min Δ 0.002 (TLOB's early stopping) | 10 | CE |
| deepofi_lstm | 100 | order flow, winsorised + z | Adam 1e-5 | 256 | val loss, patience 5 | 50 | MSE |
| siamese_lstm | 50 | order flow, winsorised + z | Adam 1e-4, wd 1e-3 | 256 | val MAE, patience 5 | 50 (compute cap) | MAE |
| custom_symmetric_lob | 128 | custom 62-feature input + 12 slow-context features | AdamW 3e-4, EMA 0.999 | 512 | val loss, patience 8, LR plateau, clip 1.0 | 50 | class-weighted CE + 0.2 · SmoothL1 |

## Implementation choices

- **All paper models**
  - BTC snapshots replace each paper's data.
  - One shared target, horizon set and sample set.
  - One chronological split.
  - Three seeds (42, 43, 44)
- **DeepLOB**
  - Inputs use the train-fitted z-score from TLOB's BTC code. The paper's rolling previous-5-day z-score isn't possible for the first training days.
  - No Dropout at evaluation.
  - Epoch cap of 100.
- **Multi-horizon DeepLOB**
  - Uses the benchmark's horizons instead of the FI-2010 ones, and the DeepLOB z-score instead of FI-2010's decimal-precision normalisation.
- **TLOB**
  - BiN keeps its mixing weights positive with softplus and guards both standard deviations. 
  - θ is fitted on training data. 
  - Uses the benchmark target instead of eq. 5–7.
  - No halving of learning rate on every epoch (improves validation loss by less than 0.002).
- **Deep OFI**
  - Order flow comes from 250 ms snapshot differences, not every book update (the file has no event stream).
  - Uses the benchmark horizons instead of multiples of Δt (1.54 s here, see the data report).
  - Uses the benchmark target instead of the point price change in dollars.
  - No 10 ms latency buffer, which can't be represented at 250 ms.
  - No open/close trimming, since BTC trades 24/7.
- **Siamese**
  - Inputs are winsorised and z-scored per column as in Kolm, since the paper specifies no order-flow normalisation. Bid and ask level i therefore get separate statistics before the shared encoder.
  - The paper leaves the head's hidden width open; it is 64 here.
  - The target cap is the 99.9% quantile of training |returns|, because the paper's +/- 1 CNY cap has no meaning for BTC.
  - Targets are proportional returns, scaled by the cap.
  - Order flow uses Kolm's −v(t−1) for price drops.
- **Custom:** It combines:
  - a shared bid/ask side encoder with exact mirror averaging (Siamese);
  - order flow (Kolm, Cont), log volumes, and prices as tick distances from the mid-price plus the mid-price change and spread in ticks (after Tsantekidis 2020, whose features are proportional and use cumulative depth);
  - class-weighted cross-entropy (Tsantekidis, eq. 19);
  - multi-horizon decoding (Zhang & Zohren) and a joint return head;
  - causal dilated time convolutions (dilations 1–32, covering the 128-snapshot window);
  - an EMA of the weights with TLOB's warm-up.
  - slow context: exponentially weighted averages with half-lives of 30 s, 2 min and 10 min of the mid change, the best-level order-flow imbalance, realised volatility and the share of snapshots where the mid moves, reset at every gap and day. They reach beyond the 32 s window: volatility clusters, and order flow persists for minutes as large orders are split (Lillo & Farmer 2004). Signed columns flip under the bid/ask mirror.

## Sources

- **Papers:**
  - Zhang, Zohren & Roberts 2019, DeepLOB: [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
  - Zhang & Zohren 2021, multi-horizon forecasting: [arXiv:2105.10430](https://arxiv.org/abs/2105.10430)
  - Berti & Kasneci 2025, TLOB: [arXiv:2502.15757](https://arxiv.org/abs/2502.15757)
  - Kolm, Turiel & Westray 2023, Deep OFI: [doi:10.1111/mafi.12413](https://doi.org/10.1111/mafi.12413)
  - Yang, Fang, Zhang & Zhou 2025, Siamese LOB: [arXiv:2505.22678](https://arxiv.org/abs/2505.22678)
  - Tsantekidis et al. 2020, stationary LOB features: [arXiv:1810.09965](https://arxiv.org/abs/1810.09965)
  - Cont, Kukanov & Stoikov 2014, order-flow imbalance: [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
  - Xu, Gould & Howison 2019, multi-level OFI: [arXiv:1907.06230](https://arxiv.org/abs/1907.06230)
  - LiT 2025, for the target protocol: [doi:10.3389/frai.2025.1616485](https://doi.org/10.3389/frai.2025.1616485)
  - Ideas for the custom model only: Lillo & Farmer 2004, order-flow persistence ([arXiv:cond-mat/0311053](https://arxiv.org/abs/cond-mat/0311053)); Stoikov 2018, micro-price ([doi:10.1080/14697688.2018.1489139](https://doi.org/10.1080/14697688.2018.1489139)); Lucchese, Pakkanen & Veraart 2024, volume representation ([arXiv:2211.13777](https://arxiv.org/abs/2211.13777))
- **Code:**
  - [zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) at `ff14d7c`
  - [zcakhaa/Multi-Horizon-Forecasting-for-Limit-Order-Books](https://github.com/zcakhaa/Multi-Horizon-Forecasting-for-Limit-Order-Books) at `1c06e35`
  - [LeonardoBerti00/TLOB](https://github.com/LeonardoBerti00/TLOB) at `f1c0af4`
- **Data:** [Kaggle: Bitcoin perpetual BTCUSDT.P limit order book](https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data).
