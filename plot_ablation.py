from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------
# Settings
# ----------------------------------------------------

BAND = "110-130"
ROOT = Path(f"runs_dct_ablation2_{BAND}")

# BASELINE_ROOT = Path("runs_dct_ablation2_{BAND}/baseline")

SETS = [
    ("add20low", 20),
    ("add40low", 40),
    ("add60low", 60),
    ("add80low", 80),
    ("add100low", 100),
    ("baseline", 0)
]

SEEDS = [0, 1]


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def load_loss_curve(run_dir):
    """
    Load metrics/loss.dat.

    Expected columns:
        iteration, epoch, train_loss, validation_loss

    Returns
    -------
    dict or None
        {
            "iteration": np.ndarray,
            "epoch": np.ndarray,
            "train": np.ndarray,
            "val": np.ndarray,
        }
    """
    loss_file = run_dir / "metrics" / "loss.dat"

    if not loss_file.exists():
        print("Missing:", loss_file)
        return None

    try:
        data = np.loadtxt(
            loss_file,
            delimiter=",",
            skiprows=1,
        )
    except (ValueError, OSError) as exc:
        print(f"Could not load {loss_file}: {exc}")
        return None

    if data.size == 0:
        print("Empty:", loss_file)
        return None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 4:
        print(
            f"Expected at least four columns in {loss_file}, "
            f"but found {data.shape[1]}"
        )
        return None

    iteration = data[:, 0]
    epoch = data[:, 1]
    train_loss = data[:, -2]
    val_loss = data[:, -1]

    finite = (
        np.isfinite(iteration)
        & np.isfinite(epoch)
        & np.isfinite(train_loss)
        & np.isfinite(val_loss)
    )

    iteration = iteration[finite]
    epoch = epoch[finite]
    train_loss = train_loss[finite]
    val_loss = val_loss[finite]

    if len(epoch) == 0:
        print("No finite loss values:", loss_file)
        return None

    # Sort by epoch in case the rows are out of order.
    order = np.argsort(epoch)

    return {
        "iteration": iteration[order],
        "epoch": epoch[order],
        "train": train_loss[order],
        "val": val_loss[order],
    }


def load_best_val_loss(run_dir):
    """Return the minimum validation loss for one run."""
    curve = load_loss_curve(run_dir)

    if curve is None:
        return None

    return float(np.min(curve["val"]))


def get_baseline_run_dirs():
    """
    Support either:

        BASELINE_ROOT/metrics/loss.dat

    or:

        BASELINE_ROOT/metrics/loss.dat
        BASELINE_ROOT/metrics/loss.dat
    """
    single_file = ROOT / "metrics" / "loss.dat"

    if single_file.exists():
        return [ROOT]

    return [
        ROOT.parent / f"{ROOT.name}/baseline_seed{seed}"
        for seed in SEEDS
    ]


def interpolate_curves(curves, max_epoch=None, points_per_epoch=10):
    """
    Interpolate multiple runs onto a common epoch grid.

    This allows mean and standard deviation curves to be computed even if
    runs were logged at slightly different epochs or ended at different times.
    """
    valid_curves = [
        curve for curve in curves
        if curve is not None and len(curve["epoch"]) > 1
    ]

    if not valid_curves:
        return None

    # Only average across the epoch interval shared by every run.
    start_epoch = max(curve["epoch"].min() for curve in valid_curves)
    end_epoch = min(curve["epoch"].max() for curve in valid_curves)

    if max_epoch is not None:
        end_epoch = min(end_epoch, max_epoch)

    if end_epoch <= start_epoch:
        return None

    number_of_points = max(
        2,
        int((end_epoch - start_epoch) * points_per_epoch) + 1,
    )

    common_epochs = np.linspace(
        start_epoch,
        end_epoch,
        number_of_points,
    )

    interpolated_val = []
    interpolated_train = []

    for curve in valid_curves:
        # Remove repeated epoch values before interpolation.
        unique_epochs, unique_indices = np.unique(
            curve["epoch"],
            return_index=True,
        )

        val = curve["val"][unique_indices]
        train = curve["train"][unique_indices]

        interpolated_val.append(
            np.interp(common_epochs, unique_epochs, val)
        )
        interpolated_train.append(
            np.interp(common_epochs, unique_epochs, train)
        )

    interpolated_val = np.asarray(interpolated_val)
    interpolated_train = np.asarray(interpolated_train)

    return {
        "epoch": common_epochs,
        "val_mean": interpolated_val.mean(axis=0),
        "val_std": interpolated_val.std(axis=0),
        "train_mean": interpolated_train.mean(axis=0),
        "train_std": interpolated_train.std(axis=0),
        "n": len(valid_curves),
    }


def plot_combined_validation_curves(
    grouped_curves,
    output_file,
    max_epoch=None,
):
    """
    Plot mean validation loss for all channel sets on one figure.

    Shaded regions show ±1 standard deviation across seeds.
    """
    plt.figure(figsize=(9, 6))

    plotted_anything = False

    for label, curves in grouped_curves.items():
        summary = interpolate_curves(
            curves,
            max_epoch=max_epoch,
        )

        if summary is None:
            print(f"Could not create combined curve for {label}")
            continue

        epochs = summary["epoch"]
        mean = summary["val_mean"]
        std = summary["val_std"]

        line, = plt.plot(
            epochs,
            mean,
            linewidth=2,
            label=f"{label} (n={summary['n']})",
        )

        if summary["n"] > 1:
            plt.fill_between(
                epochs,
                mean - std,
                mean + std,
                alpha=0.18,
                color=line.get_color(),
            )

        plotted_anything = True

    if not plotted_anything:
        plt.close()
        raise RuntimeError("No valid curves were available to plot.")

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")

    if max_epoch is None:
        plt.title(f"PCC validation curves ({BAND} Hz)")
    else:
        plt.title(
            f"PCC validation curves through epoch {max_epoch} "
            f"({BAND} Hz)"
        )
        plt.xlim(0, max_epoch)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.show()

    print("Saved:", output_file)


# ----------------------------------------------------
# Load baseline
# ----------------------------------------------------

baseline_run_dirs = get_baseline_run_dirs()

baseline_vals = []
baseline_curves = []

for run_dir in baseline_run_dirs:
    curve = load_loss_curve(run_dir)

    if curve is None:
        continue

    baseline_curves.append(curve)
    baseline_vals.append(float(np.min(curve["val"])))

if len(baseline_vals) == 0:
    raise RuntimeError("No valid baseline loss.dat files were found.")


# ----------------------------------------------------
# Load ablation runs
# ----------------------------------------------------

sizes = [0]
means = [float(np.mean(baseline_vals))]
stds = [float(np.std(baseline_vals))]

# Store complete curves for the two additional figures.
grouped_curves = {
    "Baseline": baseline_curves,
}

print(
    f"Baseline: mean={means[0]:.6f}, "
    f"std={stds[0]:.6f}, "
    f"n={len(baseline_vals)}, "
    f"values={baseline_vals}"
)

for name, size in SETS:
    vals = []
    curves = []

    for seed in SEEDS:
        run_dir = ROOT / f"{name}_seed{seed}"
        curve = load_loss_curve(run_dir)

        if curve is None:
            continue

        curves.append(curve)
        vals.append(float(np.min(curve["val"])))

    if len(vals) == 0:
        print(f"Skipping {name}: no valid runs found")
        continue

    mean = float(np.mean(vals))
    std = float(np.std(vals))

    sizes.append(size)
    means.append(mean)
    stds.append(std)

    grouped_curves[f"+{size} channels"] = curves

    print(
        f"{name}: mean={mean:.6f}, "
        f"std={std:.6f}, "
        f"n={len(vals)}, "
        f"values={vals}"
    )


# ----------------------------------------------------
# Sort ablation points by channel count
# ----------------------------------------------------

order = np.argsort(sizes)

sizes = np.asarray(sizes)[order]
means = np.asarray(means)[order]
stds = np.asarray(stds)[order]


# ----------------------------------------------------
# Plot 1: best validation loss by channel count
# ----------------------------------------------------

plt.figure(figsize=(7, 5))

plt.errorbar(
    sizes,
    means,
    yerr=stds,
    marker="o",
    linewidth=2,
    capsize=5,
    label="Per-channel CNN",
)

plt.xticks(sizes)

plt.xlabel("Additional witness channels")
plt.ylabel("Best validation loss")
# plt.ylim(0.725, 0.735)
plt.title(f"DCT channel ablation ({BAND} Hz)")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_file = f"dct_ablation_zoomout_{BAND}.png"
plt.savefig(output_file, dpi=300)
plt.show()

print("Saved:", output_file)


# ----------------------------------------------------
# Plot 2: all validation curves
# ----------------------------------------------------

plot_combined_validation_curves(
    grouped_curves=grouped_curves,
    output_file=f"dct_ablation_curves_{BAND}.png",
    max_epoch=None,
)


# ----------------------------------------------------
# Plot 3: all validation curves through epoch 20
# ----------------------------------------------------

plot_combined_validation_curves(
    grouped_curves=grouped_curves,
    output_file=f"dct_ablation_curves_epoch20_{BAND}.png",
    max_epoch=20,
)