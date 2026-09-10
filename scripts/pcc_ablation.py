from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


root = Path("runs_channel_ablation_pcc")

runs = {
    "Original (35)": [
        root / "runs_pcc_350" / "metrics" / "loss.dat",
        root / "runs_pcc_351" / "metrics" / "loss.dat",
        root / "runs_pcc_352" / "metrics" / "loss.dat",
    ],
    "+40 Low": [
        root / "add40low_seed0" / "metrics" / "loss.dat",
        root / "add40low_seed1" / "metrics" / "loss.dat",
        root / "add40low_seed2" / "metrics" / "loss.dat",
    ],
    "+80 Low": [
        root / "add80low_seed0" / "metrics" / "loss.dat",
        root / "add80low_seed1" / "metrics" / "loss.dat",
        root / "add80low_seed2" / "metrics" / "loss.dat",
    ],
    "+150 Low": [
        root / "runs_pcc_150_0" / "metrics" / "loss.dat",
        root / "runs_pcc_150_1" / "metrics" / "loss.dat",
        root / "runs_pcc_150_2" / "metrics" / "loss.dat",
    ],
}


def load_loss(path):
    data = np.loadtxt(
        path,
        delimiter=",",
        comments="#",
    )

    epochs = data[:, 1]
    train = data[:, 2]
    test = data[:, 3]

    return epochs, train, test


plt.figure(figsize=(9, 6))

for label, files in runs.items():
    loaded = [load_loss(path) for path in files]

    min_len = min(len(test) for _, _, test in loaded)

    test_curves = np.vstack([
        test[:min_len]
        for _, _, test in loaded
    ])

    epochs = loaded[0][0][:min_len]

    test_mean = test_curves.mean(axis=0)
    test_std = test_curves.std(axis=0)

    plt.plot(
        epochs,
        test_mean,
        linewidth=2,
        label=label,
    )

    plt.fill_between(
        epochs,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2,
    )

plt.xlabel("Epoch")
plt.ylabel("Test loss")
plt.title("PCC channel ablation")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

out = Path("pcc_channel_ablation.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
print("Saved:", out.resolve())

plt.show()