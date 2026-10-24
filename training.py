import functools
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel

from data import DAY_NS, HORIZONS, classify, features, load_book, samples
from models import CONFIGS

RUNS = Path("runs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH = 1024


def target_scaler(returns, task):
    """Return (scale, unscale) for the model's regression target."""
    lower, upper = np.full(returns.shape[1], -np.inf), np.full(returns.shape[1], np.inf)
    mean, std = np.zeros(returns.shape[1]), np.ones(returns.shape[1])
    if task == "mse":  # Kolm et al.: winsorise, then z-score
        lower, upper = np.quantile(returns, (0.005, 0.995), axis=0)
        clipped = np.clip(returns, lower, upper)
        mean, std = clipped.mean(0), np.maximum(clipped.std(0), 1e-8)
    elif task == "mae":  # Siamese eq. 19: cap rare jumps at the 99.9% quantile
        std = np.maximum(np.quantile(np.abs(returns), 0.999, axis=0), 1e-8)
        lower, upper = -std, std
    elif task == "joint":
        std = np.maximum(returns.std(0), 1e-8)
    return lambda r: (np.clip(r, lower, upper) - mean) / std, lambda z: z * std + mean


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False  # z-scored prices differ by ~1e-4: keep full fp32
    torch.backends.cudnn.allow_tf32 = False


def ema_with_warmup(decay):
    @torch.no_grad()
    def update(averaged, current, n):  # torch_ema's warm-up, as in upstream TLOB: decay ramps up as (1 + n) / (10 + n)
        weight = 1 - torch.clamp((1 + n) / (10 + n), max=decay)
        for a, c in zip(averaged, current):
            a.lerp_(c, weight)
    return update


@functools.cache
def feature_tensor(representation):
    return torch.from_numpy(features(representation)).to(DEVICE)


def setup(config, columns):
    """Splits as device tensors (end, label, scaled return), class weights and the window gatherer."""
    splits, _ = samples()
    train_returns = splits["train"][2][:, columns]
    scale, unscale = target_scaler(train_returns, config["task"])
    data = {split: tuple(torch.from_numpy(a).to(DEVICE) for a in (end, label[:, columns], scale(ret[:, columns]).astype(np.float32)))
            for split, (end, label, ret) in splits.items()}
    if config["class_weighted"]:
        counts = np.stack([np.maximum(np.bincount(splits["train"][1][:, h], minlength=3), 1) for h in columns])
        weights = torch.tensor(len(train_returns) / (3 * counts), dtype=torch.float32)
    else:
        weights = torch.ones(len(columns), 3)
    x, offsets = torch.cat([feature_tensor(r) for r in config["features"].split("+")], dim=1), torch.arange(1 - config["lookback"], 1, device=DEVICE)
    return data, weights.to(DEVICE), lambda end: x[end[:, None] + offsets], unscale


def unpack(output, config):
    if config["task"] == "joint":
        return output
    if config["task"] == "classification":
        return (output if config["multi"] else output[:, None]), None
    return None, output


def loss_terms(logits, predicted, labels, targets, config, weights):
    horizons = targets.shape[1]
    numerator, denominator, regression = targets.new_zeros(horizons), targets.new_ones(horizons), targets.new_zeros(horizons)
    if logits is not None:
        w = weights[torch.arange(horizons, device=DEVICE), labels]
        numerator = (F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), reduction="none").view_as(w) * w).sum(0)
        denominator = w.sum(0)
    if predicted is not None:
        error = {"mae": F.l1_loss, "mse": F.mse_loss, "joint": F.smooth_l1_loss}[config["task"]]
        regression = error(predicted, targets, reduction="none").sum(0)
    return numerator, denominator, regression


def combine(numerator, denominator, regression, count, config):
    return (numerator / denominator).mean() + config["auxiliary_weight"] * regression.mean() / count


def train_epoch(model, ema, optimizer, get, train, config, weights, generator):
    model.train()
    batch_norm = any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules())
    end, labels, targets = train
    totals, count = torch.zeros(3, len(weights), device=DEVICE), 0
    order = torch.randperm(len(end), generator=generator, device=DEVICE)
    for index in order.split(config["batch_size"]):
        if batch_norm and len(index) == 1:
            break
        terms = loss_terms(*unpack(model(get(end[index])), config), labels[index], targets[index], config, weights)
        optimizer.zero_grad(set_to_none=True)
        combine(*terms, len(index), config).backward()
        if config["gradient_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        if ema:
            ema.update_parameters(model)
        totals += torch.stack(terms).detach()
        count += len(index)
    return float(combine(*totals, count, config))


@torch.inference_mode()
def predict(model, get, end, config):
    model.eval()
    logits, returns = zip(*(unpack(model(get(index)), config) for index in end.split(EVAL_BATCH)))
    return tuple(None if part[0] is None else torch.cat(part) for part in (logits, returns))


def fit(name, columns, seed, path):
    config = CONFIGS[name]
    seed_everything(seed)
    data, weights, get, _ = setup(config, columns)
    seed_everything(seed)
    model = config["build"](len(columns)).to(DEVICE)
    ema = None
    if config["ema_decay"]:
        ema = AveragedModel(model, multi_avg_fn=ema_with_warmup(config["ema_decay"]), use_buffers=True)
        ema.update_parameters(model)  # count the initial weights as update 0, like torch_ema
    evaluated = ema.module if ema else model  # TLOB evaluates and keeps the EMA weights
    optimizer = config["optimizer"]((p for p in model.parameters() if p.requires_grad), lr=config["lr"], eps=config["eps"],
                                    weight_decay=config["weight_decay"])
    plateau = config["plateau_patience"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=plateau, min_lr=1e-6) if plateau is not None else None
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    history, best, bad, best_epoch, state = [], math.inf, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        tick = time.perf_counter()
        train_loss = train_epoch(model, ema, optimizer, get, data["train"], config, weights, generator)
        end, labels, targets = data["validation"]
        logits, predicted = predict(evaluated, get, end, config)
        loss = float(combine(*loss_terms(logits, predicted, labels, targets, config, weights), len(end), config))
        accuracy = float((logits.argmax(-1) == labels).float().mean()) if logits is not None else float("nan")
        if scheduler:
            scheduler.step(loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": loss, "validation_accuracy": accuracy, "lr": lr})
        print(f"{name} seed{seed} epoch {epoch}: train {train_loss:.6f} validation {loss:.6f} accuracy {accuracy:.4f} "
              f"lr {lr:.1e} {time.perf_counter() - tick:.0f}s", flush=True)
        score = loss if config["monitor"] == "loss" else -accuracy
        improved = score < best
        bad = 0 if improved and best - score >= config["min_delta"] else bad + 1  # patience resets only on a real improvement
        best = min(best, score)
        if improved:
            best_epoch, state = epoch, {k: v.detach().cpu().clone() for k, v in evaluated.state_dict().items()}
        if config["patience"] is not None and bad >= config["patience"]:
            break
    torch.save(state, path)
    return history, best_epoch, (time.perf_counter() - started) / 60


def test(name, columns, seed, state):
    """Test-set rows, one per horizon, for a trained state_dict."""
    config = CONFIGS[name]
    data, _, get, unscale = setup(config, columns)
    model = config["build"](len(columns)).to(DEVICE)
    model.load_state_dict(state)
    logits, predicted = predict(model, get, data["test"][0], config)
    splits, theta = samples()
    _, label, actual = splits["test"]
    timestamps = load_book()[1]
    days = (timestamps[splits["test"][0]] // DAY_NS).astype("datetime64[D]").astype(str)
    if logits is not None:
        logits = logits.cpu().numpy()
    if predicted is not None:
        predicted = unscale(predicted.cpu().numpy().astype(np.float64))
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = []
    for j, column in enumerate(columns):
        target, actual_j = label[:, column], actual[:, column]
        if logits is not None:
            probabilities = torch.from_numpy(logits[:, j]).softmax(-1).numpy()
            classes, score = logits[:, j].argmax(-1), probabilities[:, 2] - probabilities[:, 0]
        else:
            classes, score = classify(predicted[:, j], theta[column]), predicted[:, j]
        row = {"model": name, "horizon": HORIZONS[column], "seed": seed,
               "accuracy": accuracy_score(target, classes), "balanced_accuracy": balanced_accuracy_score(target, classes),
               "macro_f1": f1_score(target, classes, labels=[0, 1, 2], average="macro", zero_division=0)}
        row["pearson"], row["spearman"] = pearsonr(score, actual_j).statistic, spearmanr(score, actual_j).statistic
        if predicted is not None:
            residual = actual_j - predicted[:, j]
            if logits is not None:
                row["return_pearson"] = pearsonr(predicted[:, j], actual_j).statistic
            row.update(mae=np.abs(residual).mean(), rmse=np.sqrt(np.square(residual).mean()),
                       r2_os=1 - np.square(residual).sum() / np.square(actual_j - actual_j.mean()).sum())  # Kolm eq. 18: benchmark is the test mean
        row["parameters"] = parameters
        row["per_day"] = {day: {"macro_f1": f1_score(target[days == day], classes[days == day], labels=[0, 1, 2], average="macro", zero_division=0),
                                "pearson": pearsonr(score[days == day], actual_j[days == day]).statistic} for day in np.unique(days)}
        rows.append(row)
    return rows


def train(name, seeds=(42, 43, 44)):
    """Test rows for every seed and horizon group: loaded if already trained, otherwise trained and saved first."""
    groups = [("multi", list(range(len(HORIZONS))))] if CONFIGS[name]["multi"] else [(f"h{h}", [i]) for i, h in enumerate(HORIZONS)]
    rows = []
    for seed in seeds:
        for group, columns in groups:
            stem = RUNS / name / f"{group}-seed{seed}"
            if not stem.with_suffix(".json").exists():
                stem.parent.mkdir(parents=True, exist_ok=True)
                history, best_epoch, minutes = fit(name, columns, seed, stem.with_suffix(".pt"))
                result = test(name, columns, seed, torch.load(stem.with_suffix(".pt")))
                for row in result:
                    row.update(best_epoch=best_epoch, epochs=len(history), minutes=minutes)
                stem.with_suffix(".json").write_text(json.dumps({"rows": result, "history": history}))
            rows += json.loads(stem.with_suffix(".json").read_text())["rows"]
    return pd.DataFrame(rows)
