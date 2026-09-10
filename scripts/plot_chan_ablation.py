import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUN_ROOT = Path("runs_channel_sweep")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2]
JSON_NAME = "dc_run.json"

CONDITIONS = [
    "add40high",
    "add40low",
    "add40mixed",
    "add60high",
    "add60low",
    "add60mixed",
    "add80high",
    "add80low",
    "add80mixed",
    "add150high",
    "add150low",
    "add150mixed",
    "add20high",
    "add20low",
    "add20mixed",
    "add400high",
    "add400low",
    "add400mixed",
]


def get_loss(history):
    for key in ["test_loss", "val_loss", "valid_loss", "loss"]:
        if key in history:
            return np.array(history[key], dtype=float), key
    raise KeyError(f"No validation/test loss key found. Available keys: {history.keys()}")


def parse_channel_delta(condition):
    """
    Returns x-axis value:
      add20high -> +20
      minus5low -> -5
      ogchannels -> 0
    """
    if condition == "ogchannels":
        return 0

    m = re.search(r"add(\d+)", condition)
    if m:
        return int(m.group(1))

    m = re.search(r"minus(\d+)", condition)
    if m:
        return -int(m.group(1))

    raise ValueError(f"Could not parse channel delta from condition: {condition}")


def parse_strategy(condition):
    """
    Returns strategy label:
      add20high -> high
      minus10mixed -> mixed
      ogchannels -> baseline
    """
    if condition == "ogchannels":
        return "baseline"

    for strategy in ["high", "low", "mixed"]:
        if strategy in condition:
            return strategy

    return "unknown"


rows = []
curves = {}

for condition in CONDITIONS:
    curves[condition] = []

    for seed in SEEDS:
        run_dir = RUN_ROOT / f"{condition}_seed{seed}"
        json_path = run_dir / JSON_NAME

        if not json_path.exists():
            print(f"Missing: {json_path}")
            continue

        with open(json_path) as f:
            run = json.load(f)

        history = run["history"]
        loss, loss_key = get_loss(history)

        curves[condition].append(loss)

        rows.append({
            "condition": condition,
            "strategy": parse_strategy(condition),
            "channel_delta": parse_channel_delta(condition),
            "seed": seed,
            "loss_key": loss_key,
            "best_loss": float(loss.min()),
            "best_epoch": int(loss.argmin() + 1),
            "final_loss": float(loss[-1]),
            "n_epochs": len(loss),
            "json_path": str(json_path),
        })


df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No runs found. Check RUN_ROOT, CONDITIONS, SEEDS, and JSON_NAME.")

df.to_csv(OUT_DIR / "channel_ablation_summary_by_seed.csv", index=False)

summary = (
    df.groupby(["condition", "strategy", "channel_delta"])
      .agg(
          mean_best_loss=("best_loss", "mean"),
          std_best_loss=("best_loss", "std"),
          mean_final_loss=("final_loss", "mean"),
          std_final_loss=("final_loss", "std"),
          mean_best_epoch=("best_epoch", "mean"),
          n_seeds=("seed", "count"),
      )
      .reset_index()
      .sort_values(["strategy", "channel_delta"])
)

summary.to_csv(OUT_DIR / "channel_ablation_summary.csv", index=False)

print("\nPer-seed results:")
print(df[["condition", "strategy", "channel_delta", "seed", "best_loss", "best_epoch", "final_loss"]])

print("\nSummary:")
print(summary)


# Plot 1: performance vs channels added/subtracted
plt.figure(figsize=(9, 6))

for strategy in ["high", "low", "mixed"]:
    sub = summary[summary["strategy"] == strategy].sort_values("channel_delta")

    if sub.empty:
        continue

    plt.errorbar(
        sub["channel_delta"],
        sub["mean_best_loss"],
        yerr=sub["std_best_loss"],
        marker="o",
        capsize=5,
        linewidth=2,
        label=strategy,
    )

# Add baseline as x=0 reference, if present
baseline = summary[summary["strategy"] == "baseline"]
if not baseline.empty:
    b = baseline.iloc[0]
    plt.errorbar(
        [0],
        [b["mean_best_loss"]],
        yerr=[b["std_best_loss"]],
        marker="s",
        capsize=5,
        linewidth=2,
        label="original baseline",
    )
    plt.axhline(b["mean_best_loss"], linestyle="--", alpha=0.5)

plt.xlabel("Channels added/subtracted from original DeepClean channel set")
plt.ylabel("Best validation/test loss")
plt.title("Channel ablation performance: mean ± std over seeds")
plt.legend(title="Selection strategy")
plt.tight_layout()
plt.savefig(OUT_DIR / "channel_ablation_best_loss_vs_channels.png", dpi=200)
plt.show()


# Plot 2: same data, normalized relative to original baseline
if not baseline.empty:
    baseline_loss = baseline.iloc[0]["mean_best_loss"]

    plt.figure(figsize=(9, 6))

    for strategy in ["high", "low", "mixed"]:
        sub = summary[summary["strategy"] == strategy].sort_values("channel_delta")

        if sub.empty:
            continue

        plt.errorbar(
            sub["channel_delta"],
            sub["mean_best_loss"] - baseline_loss,
            yerr=sub["std_best_loss"],
            marker="o",
            capsize=5,
            linewidth=2,
            label=strategy,
        )

    plt.axhline(0, linestyle="--", alpha=0.5)
    plt.xlabel("Channels added/subtracted from original DeepClean channel set")
    plt.ylabel("Best loss minus original baseline")
    plt.title("Channel ablation performance relative to original baseline")
    plt.legend(title="Selection strategy")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "channel_ablation_best_loss_relative_to_baseline.png", dpi=200)
    plt.show()


# Optional: learning curves, still useful for diagnosing overfitting
plt.figure(figsize=(9, 6))

for condition, vals in curves.items():
    if len(vals) == 0:
        continue

    min_len = min(len(v) for v in vals)
    vals = np.stack([v[:min_len] for v in vals], axis=0)

    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    epochs = np.arange(1, min_len + 1)

    plt.plot(epochs, mean, label=f"{condition} (n={len(vals)})")
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.15)

plt.xlabel("Epoch")
plt.ylabel("Validation/Test Loss")
plt.title("Channel ablation learning curves: mean ± std")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "channel_ablation_curves_mean_std.png", dpi=200)
plt.show()