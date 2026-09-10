import pandas as pd

old_df = pd.read_csv("figures/channel_ablation_summary_by_seed_old.csv")
new_df = pd.read_csv("figures/channel_ablation_summary_by_seed.csv")

df = pd.concat([old_df, new_df], ignore_index=True)

# remove duplicates if any conditions were rerun
df = df.drop_duplicates(
    subset=["condition", "strategy", "channel_delta", "seed"],
    keep="last",
)

# save merged per-seed table
df.to_csv(
    "figures/channel_ablation_summary_by_seed_combined.csv",
    index=False,
)

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
)

# save merged summary table
summary.to_csv(
    "figures/channel_ablation_summary_combined.csv",
    index=False,
)

print(summary.sort_values(["strategy", "channel_delta"]))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = OUT_DIR / "channel_ablation_summary_combined.csv"
summary = pd.read_csv(SUMMARY_CSV)

# keep only add/baseline by default; remove this if you want minus5 included too
# summary = summary[summary["condition"].str.startswith("add") | (summary["strategy"] == "baseline")]

strategy_order = ["high", "low", "mixed"]
summary = summary.sort_values(["strategy", "channel_delta"])


# Plot 1: best loss vs channel delta
plt.figure(figsize=(9, 6))

for strategy in strategy_order:
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
plt.ylabel("Best validation loss")
plt.title("Channel ablation performance: mean ± std over seeds")
plt.legend(title="Selection strategy")
plt.tight_layout()
plt.savefig(OUT_DIR / "channel_ablation_best_loss_vs_channels_combined.png", dpi=200)
plt.show()


# Plot 2: relative to baseline
if not baseline.empty:
    baseline_loss = baseline.iloc[0]["mean_best_loss"]

    plt.figure(figsize=(9, 6))

    for strategy in strategy_order:
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
    plt.savefig(OUT_DIR / "channel_ablation_best_loss_relative_combined.png", dpi=200)
    plt.show()