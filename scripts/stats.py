import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# settings
# ----------------------------
results_dir = "train_dir"
pattern = "add5chan*.json"   # +5 channels
pattern2 = "dc_run*.json"   #  DeepClean - transform
pattern3 = "dcadd10*.json"
pattern4 = "dcadd20*.json"

save_plot = True 
plot_path = os.path.join(results_dir, "dcaddchan_multiseed_plot.png")

# ----------------------------
# helper to load runs
# ----------------------------
def load_runs(file_list):
    train_histories = []
    val_histories = []

    for fpath in file_list:
        with open(fpath, "r") as f:
            run = json.load(f)

        train_histories.append(run["history"]["train_loss"])
        val_histories.append(run["history"]["val_loss"])

    train_arr = np.array(train_histories)
    val_arr = np.array(val_histories)

    return train_arr, val_arr

# ----------------------------
# load +10 channel runs
# ----------------------------
files4 = sorted(glob.glob(os.path.join(results_dir, pattern4)))

if not files4:
    raise FileNotFoundError(f"No files found for pattern3: {pattern4}")

print("Found +20 channel files:")
for f in files4:
    print(" ", f)

train_arr4, val_arr4 = load_runs(files4)

n_seeds4, n_epochs4 = train_arr4.shape
epochs4 = np.arange(1, n_epochs4 + 1)

train_mean4 = train_arr4.mean(axis=0)
train_std4 = train_arr4.std(axis=0)

val_mean4 = val_arr4.mean(axis=0)
val_std4 = val_arr4.std(axis=0)


# ----------------------------
# load +10 channel runs
# ----------------------------
files3 = sorted(glob.glob(os.path.join(results_dir, pattern3)))

if not files3:
    raise FileNotFoundError(f"No files found for pattern3: {pattern3}")

print("Found +10 channel files:")
for f in files3:
    print(" ", f)

train_arr3, val_arr3 = load_runs(files3)

n_seeds3, n_epochs3 = train_arr3.shape
epochs3 = np.arange(1, n_epochs3 + 1)

train_mean3 = train_arr3.mean(axis=0)
train_std3 = train_arr3.std(axis=0)

val_mean3 = val_arr3.mean(axis=0)
val_std3 = val_arr3.std(axis=0)

# ----------------------------
# load +5 channel runs
# ----------------------------
files = sorted(glob.glob(os.path.join(results_dir, pattern)))

if not files:
    raise FileNotFoundError(f"No files found for pattern1: {pattern}")

print("Found +5 channel files:")
for f in files:
    print(" ", f)

train_arr, val_arr = load_runs(files)

n_seeds, n_epochs = train_arr.shape
epochs = np.arange(1, n_epochs + 1)

train_mean = train_arr.mean(axis=0)
train_std = train_arr.std(axis=0)

val_mean = val_arr.mean(axis=0)
val_std = val_arr.std(axis=0)


# ----------------------------
# load original DeepClean-Transform runs
# ----------------------------
files2 = sorted(glob.glob(os.path.join(results_dir, pattern2)))

if not files2:
    print("\n⚠️ No files found for pattern2 — skipping original comparison")
    train2_mean = val2_mean = train2_std = val2_std = None
else:
    print("\nFound original DeepClean files:")
    for f in files2:
        print(" ", f)

    train_arr2, val_arr2 = load_runs(files2)

    # sanity check
    if train_arr2.shape[1] != n_epochs:
        raise ValueError("Epoch mismatch between pattern1 and pattern2")

    train2_mean = train_arr2.mean(axis=0)
    train2_std = train_arr2.std(axis=0)

    val2_mean = val_arr2.mean(axis=0)
    val2_std = val_arr2.std(axis=0)


# ----------------------------
# plot
# ----------------------------
plt.figure(figsize=(8, 5))
# ---- +20 channels ----

plt.plot(epochs, val_mean4, label="+20 channels (val)", linestyle='-')
plt.fill_between(
    epochs,
    val_mean4 - val_std4,
    val_mean4 + val_std4,
    alpha=0.2
)

# ---- +10 channels ----
plt.plot(epochs, val_mean3, label="+10 channels (val)", linestyle='-')
plt.fill_between(
    epochs,
    val_mean3 - val_std3,
    val_mean3 + val_std3,
    alpha=0.2
)

# ---- +5 channels ----
plt.plot(epochs, val_mean, label="+5 channels (val)", linestyle='-')
plt.fill_between(
    epochs,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.2
)

# ---- original DeepClean ----
if val2_mean is not None:
    plt.plot(epochs, val2_mean, label="Original (val)", linestyle='--')
    plt.fill_between(
        epochs,
        val2_mean - val2_std,
        val2_mean + val2_std,
        alpha=0.2
    )

# ----------------------------
# formatting
# ----------------------------
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DeepClean: Original vs +5+10 Low-Coherence Channels")

plt.xticks(epochs)
plt.xlim(1, n_epochs)

plt.legend()
plt.tight_layout()

# ----------------------------
# save
# ----------------------------
if save_plot:
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot to: {plot_path}")

plt.show()