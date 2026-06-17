import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUN_ROOT = Path("runs_channel_ablation_deepclean")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = [
    "ogchannels",
    "top35",
    "channelsadd5",
    "channelsminus5",
]

SEEDS = [0, 1, 2]
JSON_NAME = "dc_run.json"


def get_loss(history):
    for key in ["test_loss", "val_loss", "valid_loss", "loss"]:
        if key in history:
            return np.array(history[key], dtype=float), key
    raise KeyError(f"No validation/test loss key found. Available keys: {history.keys()}")


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
    df.groupby("condition")
      .agg(
          mean_best_loss=("best_loss", "mean"),
          std_best_loss=("best_loss", "std"),
          mean_final_loss=("final_loss", "mean"),
          std_final_loss=("final_loss", "std"),
          mean_best_epoch=("best_epoch", "mean"),
          n_seeds=("seed", "count"),
      )
      .reset_index()
)

summary.to_csv(OUT_DIR / "channel_ablation_summary.csv", index=False)

print("\nPer-seed results:")
print(df[["condition", "seed", "best_loss", "best_epoch", "final_loss"]])

print("\nSummary:")
print(summary)


# Plot 1: mean validation/test curves ± std
plt.figure(figsize=(9, 6))

for condition in CONDITIONS:
    vals = curves.get(condition, [])

    if len(vals) == 0:
        continue

    min_len = min(len(v) for v in vals)
    vals = np.stack([v[:min_len] for v in vals], axis=0)

    mean = vals.mean(axis=0)
    std = vals.std(axis=0)

    epochs = np.arange(1, min_len + 1)

    label = f"{condition} (n={len(vals)})"
    plt.plot(epochs, mean, label=label)
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)

plt.xlabel("Epoch")
plt.ylabel("Validation/Test Loss")
plt.title("Channel-set ablation: mean learning curves ± std")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "channel_ablation_curves_mean_std.png", dpi=200)
plt.show()


# Plot 2: best loss bar plot ± std
ordered = [c for c in CONDITIONS if c in summary["condition"].values]
summary_ordered = summary.set_index("condition").loc[ordered]

plt.figure(figsize=(8, 5))
plt.bar(
    summary_ordered.index,
    summary_ordered["mean_best_loss"],
    yerr=summary_ordered["std_best_loss"],
    capsize=5,
)
plt.ylabel("Best Validation/Test Loss")
plt.title("Best loss by channel set: mean ± std over seeds")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "channel_ablation_best_loss_bar.png", dpi=200)
plt.show()


# Plot 3: individual seed curves, faint, plus mean bold
plt.figure(figsize=(9, 6))

for condition in CONDITIONS:
    vals = curves.get(condition, [])

    if len(vals) == 0:
        continue

    min_len = min(len(v) for v in vals)
    vals = np.stack([v[:min_len] for v in vals], axis=0)
    epochs = np.arange(1, min_len + 1)

    for v in vals:
        plt.plot(epochs, v, alpha=0.25)

    plt.plot(epochs, vals.mean(axis=0), linewidth=2.5, label=f"{condition} mean")

plt.xlabel("Epoch")
plt.ylabel("Validation/Test Loss")
plt.title("Channel-set ablation: individual seeds and means")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "channel_ablations.png", dpi=200)
plt.show()


"""
Per-seed results:
         condition  seed  best_loss  best_epoch  final_loss
0       ogchannels     0   0.728233          13    0.728856
1       ogchannels     1   0.728572          16    0.728678
2       ogchannels     2   0.728118          16    0.728283
3            top35     0   0.729638          13    0.730146
4            top35     1   0.729103          20    0.729103
5            top35     2   0.729523          20    0.729523
6     channelsadd5     0   0.728178          13    0.728631
7     channelsadd5     1   0.727760          17    0.727830
8     channelsadd5     2   0.728449          12    0.728921
9   channelsminus5     0   0.730102          12    0.730354
10  channelsminus5     1   0.729000          20    0.729000
11  channelsminus5     2   0.729646          14    0.729879

Summary:
        condition  mean_best_loss  std_best_loss  mean_final_loss  std_final_loss  mean_best_epoch  n_seeds
0    channelsadd5        0.728129       0.000347         0.728461        0.000565        14.000000        3
1  channelsminus5        0.729583       0.000554         0.729744        0.000687        15.333333        3
2      ogchannels        0.728308       0.000236         0.728606        0.000293        15.000000        3
3           top35        0.729421       0.000282         0.729591        0.000525        17.666667        3
"""