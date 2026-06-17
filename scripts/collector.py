import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


root = Path("runs_coherence_sweep")
baseline_file = Path("train_dir/dcrun_4.json")

rows = []


def get_losses(history):
    val_losses = (
        history.get("test_loss")
        or history.get("val_loss")
        or history.get("valid_loss")
    )

    train_losses = (
        history.get("train_loss")
        or history.get("loss")
    )

    return train_losses, val_losses


# -----------------------------
# Load baseline DeepClean result
# -----------------------------

with open(baseline_file) as f:
    baseline_data = json.load(f)

baseline_train_losses, baseline_val_losses = get_losses(
    baseline_data["history"]
)

baseline_best_val = min(baseline_val_losses)
baseline_final_val = baseline_val_losses[-1]
baseline_best_epoch = baseline_val_losses.index(baseline_best_val) + 1

print(f"Baseline best val loss: {baseline_best_val:.6f}")
print(f"Baseline final val loss: {baseline_final_val:.6f}")
print(f"Baseline best epoch: {baseline_best_epoch}")


# -----------------------------
# Load sweep runs
# -----------------------------

for json_file in root.rglob("*.json"):
    run_name = json_file.parent.name

    # skip unrelated files
    if "channel_sets" in str(json_file):
        continue

    with open(json_file) as f:
        data = json.load(f)

    if "history" not in data:
        continue

    history = data["history"]

    train_losses, val_losses = get_losses(history)

    if val_losses is None:
        print("Skipping, no val/test loss:", json_file)
        continue

    meta = data.get("channel_set_metadata", {})

    m = re.search(r"added_?(\d+)|added(\d+)", run_name)
    added_k = int(next(g for g in m.groups() if g is not None)) if m else None

    if run_name.startswith("very_low"):
        bin_name = "very_low"
    elif run_name.startswith("low"):
        bin_name = "low"
    elif run_name.startswith("medium"):
        bin_name = "medium"
    elif run_name.startswith("high"):
        bin_name = "high"
    else:
        bin_name = None

    best_val = min(val_losses)

    rows.append({
        "run": run_name,
        "json_file": str(json_file),
        "coherence_bin": bin_name,
        "added_k": added_k,
        "avg_added_coherence": meta.get("avg_added_coherence"),
        "best_val_loss": best_val,
        "final_val_loss": val_losses[-1],
        "best_epoch": val_losses.index(best_val) + 1,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "delta_vs_baseline": best_val - baseline_best_val,
    })


df = pd.DataFrame(rows)

df = df.dropna(subset=[
    "coherence_bin",
    "added_k",
    "avg_added_coherence",
    "best_val_loss",
])

df = df.sort_values(["added_k", "coherence_bin"])

df.to_csv("coherence_sweep_summary.csv", index=False)

# print(df[[
#     "run",
#     "coherence_bin",
#     "added_k",
#     "avg_added_coherence",
#     "best_val_loss",
#     "delta_vs_baseline",
#     "final_val_loss",
#     "best_epoch",
# ]])


# ---------------------------------------------------
# Plot 1: performance vs average added coherence
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

for bin_name in ["high", "medium", "low", "very_low"]:
    subset = df[df["coherence_bin"] == bin_name]

    if len(subset) == 0:
        continue

    plt.scatter(
        subset["avg_added_coherence"],
        subset["best_val_loss"],
        label=bin_name,
    )

plt.axhline(
    baseline_best_val,
    linestyle="--",
    label=f"baseline DeepClean ({baseline_best_val:.4f})",
)

plt.xlabel("Average Added Coherence")
plt.ylabel("Best Validation Loss")
plt.title("DeepClean Performance vs Added Channel Coherence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("performance_vs_added_coherence_with_baseline.png", dpi=200)
plt.close()


# ---------------------------------------------------
# Plot 2: performance vs number of added channels
# all coherence bins
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

for bin_name in ["high", "medium", "low", "very_low"]:
    subset = df[df["coherence_bin"] == bin_name]
    subset = subset.sort_values("added_k")

    if len(subset) == 0:
        continue

    plt.plot(
        subset["added_k"],
        subset["best_val_loss"],
        marker="o",
        label=bin_name,
    )

plt.axhline(
    baseline_best_val,
    linestyle="--",
    label=f"baseline DeepClean ({baseline_best_val:.4f})",
)

plt.xlabel("Number of Added Channels")
plt.ylabel("Best Validation Loss")
plt.title("DeepClean Performance vs Added Channel Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("performance_vs_channel_count_all_bins_with_baseline.png", dpi=200)
plt.close()