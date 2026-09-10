from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Settings
# ============================================================

BAND = "110-130"

MODEL_ROOTS = {
    "DeepClean": Path(f"runs_dc_ablation_{BAND}"),
    "PCC": Path(f"runs_pcc_ablation_{BAND}"),
    "DC-T": Path(f"runs_dct_ablation2_{BAND}"),
}

CHANNEL_SETS = [
    ("add20low", 20),
    ("add40low", 40),
    ("add60low", 60),
    ("add80low", 80),
    ("add100low", 100),
    ("add150low", 150),
    ("add200low", 200),
    ("add400low", 400),
]

SEEDS = [0, 1, 2]

BASELINE_PREFIX = "baseline"

OUTPUT_FILE = f"dc_pcc_dct_relative_ablation_{BAND}.png"

# Set to None for automatic y-axis limits.
Y_LIMITS = None

# Example:
# Y_LIMITS = (-0.01, 0.03)


# ============================================================
# Loss loading
# ============================================================

def load_loss_curve(run_dir):
    """
    Load a metrics/loss.dat file.

    Expected columns:
        iteration, epoch, train_loss, validation_loss

    The first row is assumed to be a header.

    Partially completed runs are supported. Every complete, finite row
    currently available in loss.dat is used.
    """
    loss_file = run_dir / "metrics" / "loss.dat"

    if not loss_file.exists():
        print(f"Missing: {loss_file}")
        return None

    try:
        data = np.genfromtxt(
            loss_file,
            delimiter=",",
            skip_header=1,
            invalid_raise=False,
        )
    except (ValueError, OSError) as exc:
        print(f"Could not load {loss_file}: {exc}")
        return None

    if data.size == 0:
        print(f"Empty: {loss_file}")
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

    if len(val_loss) == 0:
        print(f"No finite loss values: {loss_file}")
        return None

    order = np.argsort(epoch)

    return {
        "iteration": iteration[order],
        "epoch": epoch[order],
        "train": train_loss[order],
        "val": val_loss[order],
    }


def load_best_val_loss(run_dir):
    """
    Return the minimum validation loss currently available for one run.
    """
    curve = load_loss_curve(run_dir)

    if curve is None:
        return None

    return float(np.min(curve["val"]))


# ============================================================
# Run-directory helpers
# ============================================================

def get_seeded_run_dir(root, prefix, seed):
    """
    Return the expected directory for a seeded run.

    Example:
        runs_pcc_ablation_110-130/add200low_seed0
    """
    return root / f"{prefix}_seed{seed}"


def load_baselines_by_seed(root, model_name):
    """
    Load baseline validation losses.

    Preferred layout:
        root/baseline_seed0/metrics/loss.dat
        root/baseline_seed1/metrics/loss.dat
        root/baseline_seed2/metrics/loss.dat

    Alternative shared-baseline layout:
        root/metrics/loss.dat

    PCC special case:
        If PCC has exactly one seeded baseline, that baseline is shared
        across all PCC ablation seeds.

    Returns
    -------
    dict
        Mapping from ablation seed to baseline validation loss.
    """
    baselines = {}

    for seed in SEEDS:
        run_dir = get_seeded_run_dir(
            root,
            BASELINE_PREFIX,
            seed,
        )

        value = load_best_val_loss(run_dir)

        if value is not None:
            baselines[seed] = value

    # All expected seeded baselines exist.
    if len(baselines) == len(SEEDS):
        return baselines

    # Check for one unseeded baseline directly under the model root.
    single_baseline = load_best_val_loss(root)

    if single_baseline is not None:
        print(
            f"Warning: using one shared baseline for all "
            f"{model_name} seeds: {single_baseline:.8f}"
        )

        return {
            seed: single_baseline
            for seed in SEEDS
        }

    # PCC has one baseline run but multiple ablation seeds.
    if model_name == "PCC" and len(baselines) == 1:
        baseline_seed, baseline_value = next(
            iter(baselines.items())
        )

        print(
            f"Warning: PCC has only baseline_seed{baseline_seed}. "
            f"Using baseline value {baseline_value:.8f} "
            f"for every PCC ablation seed."
        )

        return {
            seed: baseline_value
            for seed in SEEDS
        }

    # Retain available seed-matched baselines for other models.
    if baselines:
        print(
            f"Warning: {model_name} has baselines only for seeds "
            f"{sorted(baselines)}. Ablation seeds without a matching "
            f"baseline will be skipped."
        )

        return baselines

    raise RuntimeError(
        f"No valid baseline was found under {root}. Expected either:\n"
        f"  {root}/{BASELINE_PREFIX}_seed<seed>/metrics/loss.dat\n"
        f"or:\n"
        f"  {root}/metrics/loss.dat"
    )


# ============================================================
# Relative ablation summary
# ============================================================

def summarize_model(root, model_name):
    """
    Compute channel-ablation changes relative to the model's baseline.

    For each seed:

        delta =
            best ablation validation loss
            - baseline validation loss

    Interpretation:
        delta < 0 means the ablation run performed better.
        delta > 0 means the ablation run performed worse.

    Partially completed runs are included using their best currently
    available validation loss.
    """
    print("\n" + "=" * 72)
    print(model_name)
    print(f"Root: {root}")
    print("=" * 72)

    baselines = load_baselines_by_seed(
        root=root,
        model_name=model_name,
    )

    print("\nBaseline values:")

    for seed in sorted(baselines):
        print(
            f"  seed {seed}: "
            f"{baselines[seed]:.8f}"
        )

    sizes = [0]
    means = [0.0]
    stds = [0.0]
    counts = [len(baselines)]

    seed_deltas = {
        0: {
            seed: 0.0
            for seed in baselines
        }
    }

    for prefix, size in CHANNEL_SETS:
        deltas = []
        runs = {}

        print(f"\n{prefix}:")

        for seed in SEEDS:
            if seed not in baselines:
                print(
                    f"  seed {seed}: "
                    f"no matching baseline; skipped"
                )
                continue

            run_dir = get_seeded_run_dir(
                root,
                prefix,
                seed,
            )

            curve = load_loss_curve(run_dir)

            if curve is None:
                continue

            ablation_loss = float(
                np.min(curve["val"])
            )

            completed_epoch = float(
                np.max(curve["epoch"])
            )

            baseline_loss = baselines[seed]
            delta = ablation_loss - baseline_loss

            deltas.append(delta)
            runs[seed] = delta

            print(
                f"  seed {seed}: "
                f"epochs={completed_epoch:g}, "
                f"baseline={baseline_loss:.8f}, "
                f"ablation={ablation_loss:.8f}, "
                f"delta={delta:+.8f}"
            )

        if not deltas:
            print(
                f"Skipping {prefix}: "
                f"no valid ablation/baseline pairs"
            )
            continue

        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas))

        sizes.append(size)
        means.append(mean_delta)
        stds.append(std_delta)
        counts.append(len(deltas))
        seed_deltas[size] = runs

        print(
            f"  summary: "
            f"mean delta={mean_delta:+.8f}, "
            f"std={std_delta:.8f}, "
            f"n={len(deltas)}"
        )

    order = np.argsort(sizes)

    return {
        "sizes": np.asarray(
            sizes,
            dtype=float,
        )[order],
        "means": np.asarray(
            means,
            dtype=float,
        )[order],
        "stds": np.asarray(
            stds,
            dtype=float,
        )[order],
        "counts": np.asarray(
            counts,
            dtype=int,
        )[order],
        "seed_deltas": seed_deltas,
    }


# ============================================================
# Plotting
# ============================================================

def plot_model_comparison(summaries):
    """
    Plot DeepClean, PCC, and DC-T channel-ablation robustness.
    """
    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )

    all_sizes = []
    plotted_anything = False

    for model_name, summary in summaries.items():
        sizes = summary["sizes"]
        means = summary["means"]
        stds = summary["stds"]

        if len(sizes) == 0:
            continue

        ax.errorbar(
            sizes,
            means,
            yerr=stds,
            marker="o",
            markersize=7,
            linewidth=2,
            capsize=6,
            capthick=1.5,
            elinewidth=1.5,
            label=model_name,
        )

        all_sizes.extend(
            sizes.tolist()
        )

        plotted_anything = True

    if not plotted_anything:
        plt.close(fig)

        raise RuntimeError(
            "No valid model summaries were available."
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.8,
        alpha=0.65,
    )

    unique_sizes = sorted(
        set(all_sizes)
    )

    ax.set_xticks(unique_sizes)

    ax.set_xlabel(
        "Channels added to original witness-channel set",
        fontsize=12,
    )

    ax.set_ylabel(
        "Best validation loss minus model baseline",
        fontsize=12,
    )

    ax.set_title(
        "Channel-ablation robustness: "
        "DeepClean vs. PCC vs. DC-T\n"
        f"{BAND} Hz",
        fontsize=14,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        title="Model",
        fontsize=11,
        title_fontsize=11,
    )

    if Y_LIMITS is not None:
        ax.set_ylim(*Y_LIMITS)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(f"\nSaved: {OUTPUT_FILE}")


# ============================================================
# Tabular summary
# ============================================================

def print_comparison_table(summaries):
    """
    Print mean, standard deviation, and number of available seeds
    for every model and channel count.
    """
    all_sizes = sorted({
        int(size)
        for summary in summaries.values()
        for size in summary["sizes"]
    })

    model_names = [
        model_name
        for model_name in MODEL_ROOTS
        if model_name in summaries
    ]

    table_width = (
        10
        + len(model_names) * (18 + 18 + 6)
    )

    print(
        "\n"
        + "=" * table_width
    )

    print(
        "RELATIVE BEST-VALIDATION-LOSS SUMMARY"
    )

    print(
        "=" * table_width
    )

    header = f"{'Channels':>10}"

    for model_name in model_names:
        header += (
            f"{model_name + ' mean':>18}"
            f"{model_name + ' std':>18}"
            f"{'n':>6}"
        )

    print(header)
    print("-" * len(header))

    for size in all_sizes:
        row = [
            f"{size:>10}"
        ]

        for model_name in model_names:
            summary = summaries.get(
                model_name
            )

            if summary is None:
                row.extend([
                    f"{'--':>18}",
                    f"{'--':>18}",
                    f"{'--':>6}",
                ])
                continue

            matches = np.where(
                summary["sizes"] == size
            )[0]

            if len(matches) == 0:
                row.extend([
                    f"{'--':>18}",
                    f"{'--':>18}",
                    f"{'--':>6}",
                ])
                continue

            index = int(matches[0])

            row.extend([
                f"{summary['means'][index]:+18.8f}",
                f"{summary['stds'][index]:18.8f}",
                f"{summary['counts'][index]:6d}",
            ])

        print("".join(row))


# ============================================================
# Main
# ============================================================

def main():
    summaries = {}

    for model_name, root in MODEL_ROOTS.items():
        if not root.exists():
            print(
                f"Warning: run directory does not exist "
                f"for {model_name}: {root}"
            )
            continue

        try:
            summaries[model_name] = summarize_model(
                root=root,
                model_name=model_name,
            )
        except RuntimeError as exc:
            print(
                f"Warning: skipping {model_name}: {exc}"
            )

    if not summaries:
        raise RuntimeError(
            "No valid DeepClean, PCC, or DC-T "
            "run directories were found."
        )

    print_comparison_table(summaries)
    plot_model_comparison(summaries)


if __name__ == "__main__":
    main()