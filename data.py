import functools
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

PROCESSED = Path("data/processed")
DAY_NS = 86_400_000_000_000
HALF_LIVES = (120, 480, 2400)
HORIZONS = (1, 2, 4, 8, 20)
MAX_LOOKBACK = 128
SPLITS = {"train": ("2023-01-09", "2023-01-17"), "validation": ("2023-01-18", "2023-01-18"), "test": ("2023-01-19", "2023-01-20")}


@functools.cache
def load_book():
    return tuple(np.load(PROCESSED / f"{name}.npy", mmap_mode="r") for name in ("book", "timestamps", "segments"))


def day_codes(split):
    start, end = SPLITS[split]
    return np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1).astype(np.int64)


def classify(returns, theta):
    return np.where(returns > theta, 2, np.where(returns < -theta, 0, 1)).astype(np.int64)


@functools.cache
def samples():
    book, timestamps, segments = load_book()
    n, reach = len(timestamps), max(HORIZONS)
    days = timestamps // DAY_NS
    block = np.cumsum(np.r_[True, (np.diff(segments) != 0) | (np.diff(days) != 0)])
    mid = np.asarray(book[:, 0]).astype(np.int64) + np.asarray(book[:, 2]).astype(np.int64)
    prefix = np.r_[0, np.cumsum(mid)]
    t = np.arange(MAX_LOOKBACK, n - reach, dtype=np.int64)
    t = t[(mid[t] != mid[t - 1]) & (block[t - MAX_LOOKBACK] == block[t + reach])]
    current = mid[t]
    returns = np.stack([(prefix[t + h + 1] - prefix[t + 1] - h * current) / (h * current) for h in HORIZONS], axis=1)
    rows = {split: np.flatnonzero(np.isin(days[t], day_codes(split))) for split in SPLITS}
    theta = np.quantile(np.abs(returns[rows["train"]]), 1 / 3, axis=0)
    return {split: (t[r], classify(returns[r], theta), returns[r]) for split, r in rows.items()}, theta


def context(book, timestamps, segments):
    """EWMAs of mid change and touch order-flow imbalance (signed), squared mid change and activity (unsigned); rows 1..N-1."""
    first = np.asarray(book[:, :4], dtype=np.float64)
    mid = (first[:, 0] + first[:, 2]) / 2
    ask_flow = (first[1:, 0] <= first[:-1, 0]) * first[1:, 1] - (first[1:, 0] >= first[:-1, 0]) * first[:-1, 1]
    bid_flow = (first[1:, 2] >= first[:-1, 2]) * first[1:, 3] - (first[1:, 2] <= first[:-1, 2]) * first[:-1, 3]
    change = np.diff(mid, prepend=mid[0])
    inputs = np.stack((change, np.r_[0, bid_flow - ask_flow], change ** 2, change != 0), axis=1)
    starts = np.r_[0, np.flatnonzero((np.diff(segments) != 0) | (np.diff(timestamps // DAY_NS) != 0)) + 1, len(mid)]
    out = np.empty((len(mid), 12))
    for a, b in zip(starts[:-1], starts[1:]):
        x = inputs[a:b].copy()
        x[0] = 0  # differences across a block boundary are meaningless
        n = np.arange(1, b - a + 1)[:, None]
        for i, half_life in enumerate(HALF_LIVES):
            decay = 0.5 ** (1 / half_life)
            y = lfilter([1 - decay], [1, -decay], x, axis=0) / (1 - decay ** n)  # bias-corrected like Adam
            out[a:b, [i, 3 + i, 6 + i, 9 + i]] = y
    out[:, 6:9] = np.log(np.sqrt(np.maximum(out[:, 6:9], 0)) + 1e-3)
    return out[1:].astype(np.float32)


def raw_features(book, representation):
    if representation == "book":
        return book[1:]
    asks, ask_volume, bids, bid_volume = (book[:, i::4] for i in range(4))
    ask_flow = (asks[1:] <= asks[:-1]) * ask_volume[1:] - (asks[1:] >= asks[:-1]) * ask_volume[:-1]
    bid_flow = (bids[1:] >= bids[:-1]) * bid_volume[1:] - (bids[1:] <= bids[:-1]) * bid_volume[:-1]
    if representation == "order_flow":
        return np.concatenate((bid_flow, ask_flow), axis=1).astype(np.float32)
    mids = (book[:, 0] + book[:, 2]) / 2
    mid = mids[1:, None]
    return np.concatenate((asks[1:] - mid, np.log1p(ask_volume[1:]), np.arcsinh(ask_flow),
                           mid - bids[1:], np.log1p(bid_volume[1:]), np.arcsinh(bid_flow),
                           np.diff(mids)[:, None], asks[1:, :1] - bids[1:, :1]), axis=1).astype(np.float32)


@functools.cache
def features(representation):
    """Normalised float32 [N, width]; row r is built from rows r - 1 and r, row 0 is zero."""
    book, timestamps, segments = load_book()
    x = context(book, timestamps, segments) if representation == "context" else raw_features(np.asarray(book), representation)
    train = np.isin(timestamps // DAY_NS, day_codes("train"))
    rows = np.flatnonzero(train[1:] & (np.diff(segments) == 0)) + 1
    rows = rows[::-(-len(rows) // 1_000_000)]
    fit = x[rows - 1].astype(np.float64)
    lower = upper = None
    if representation == "order_flow":  # Kolm et al.: winsorise, then z-score per column
        lower, upper = np.quantile(fit, (0.005, 0.995), axis=0)
        fit = np.clip(fit, lower, upper)
    mean, std = fit.mean(0), fit.std(0)
    if representation == "book":  # TLOB: one mean/std over all prices, one over all volumes
        groups = [slice(0, None, 2), slice(1, None, 2)]
    elif representation == "custom":  # matching bid/ask columns share statistics
        groups = [[a + level, b + level] for a, b in ((0, 30), (10, 40), (20, 50)) for level in range(10)]
    else:
        groups = []
    for columns in groups:
        mean[columns], std[columns] = fit[:, columns].mean(), fit[:, columns].std()
    if representation == "custom":
        mean[60] = 0.0  # keeps the mid change sign-equivariant for the mirror average
    elif representation == "context":
        mean[:6] = 0.0  # signed columns flip under the mirror
    std = np.where(std < 1e-8, 1.0, std)
    z = x.astype(np.float64)
    del x
    if lower is not None:
        np.clip(z, lower, upper, out=z)
    z -= mean
    z /= std
    if representation in ("custom", "context"):
        np.clip(z, -10, 10, out=z)
    out = np.zeros((len(book), z.shape[1]), dtype=np.float32)
    out[1:] = z
    return out
