from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# dct_correction = [
#     Path("runs_dct_baseline_110-130_correction/baseline_seed0/metrics/loss.dat"),
#     Path("runs_dct_baseline_110-130_correction/baseline_seed1/metrics/loss.dat"),
#     Path("runs_dct_baseline_110-130_correction/baseline_seed2/metrics/loss.dat"),
# ]

# dct_corr_norm = [
#     Path("runs_dct_baseline_110-130_corr_norm/baseline_seed0/metrics/loss.dat"),
#     Path("runs_dct_baseline_110-130_corr_norm/baseline_seed1/metrics/loss.dat"),
#     Path("runs_dct_baseline_110-130_corr_norm/baseline_seed2/metrics/loss.dat"),
# ]

# dct_freeze_5 = [
#     Path("runs_dct_baseline_110-130_freeze_5/baseline_seed0/metrics/loss.dat"),
#     Path("runs_dct_baseline_110-130_freeze_5/baseline_seed1/metrics/loss.dat"),
#     Path("runs_dct_baseline_110-130_freeze_5/baseline_seed2/metrics/loss.dat"),
# ]
dct_scale_025 = [
    Path("runs_dct_baseline_110-130_scale_0.25/baseline_seed0/metrics/loss.dat"),
    Path("runs_dct_baseline_110-130_scale_0.25/baseline_seed1/metrics/loss.dat"),
    Path("runs_dct_baseline_110-130_scale_0.25/baseline_seed2/metrics/loss.dat"),
]

dc_baseline_110 = [
    Path("runs_dc_baseline_110-130/baseline_seed0/metrics/loss.dat"),
    Path("runs_dc_baseline_110-130/baseline_seed1/metrics/loss.dat"),
    Path("runs_dc_baseline_110-130/baseline_seed2/metrics/loss.dat"),
]

dct_transformer_logs = [
    Path("runs_dct_baseline_110-130_logs/baseline_seed0/metrics/loss.dat"),
    Path("runs_dct_baseline_110-130_logs/baseline_seed1/metrics/loss.dat"),
    Path("runs_dct_baseline_110-130_logs/baseline_seed2/metrics/loss.dat"),
]


# dc_150 = [
#     Path("runs_dc_150_110-130_0/metrics/loss.dat"), 
#     Path("runs_dc_150_110-130_1/metrics/loss.dat"), 
#     Path("runs_dc_150_110-130_2/metrics/loss.dat"),
# ]


# pcc_35 = [
#     Path("runs_pcc_35_110-130_0/metrics/loss.dat"),
#     Path("runs_pcc_35_110-130_1/metrics/loss.dat"),
#     Path("runs_pcc_35_110-130_2/metrics/loss.dat"),
# ]

# dc_35 = [
#     Path("runs_dc_150_110-130_0/metrics/loss.dat"),
#     Path("runs_dc_150_110-130_1/metrics/loss.dat"),
#     Path("runs_dc_150_110-130_2/metrics/loss.dat"),
# ]

def load_loss(path):
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=["step", "epoch", "train", "test"],
        skipinitialspace=True,
    )

    # remove commas from strings if necessary
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(",", "")
        df[c] = pd.to_numeric(df[c])

    return df


def compute_mean_std(files):
    dfs = [load_loss(f) for f in files]

    min_len = min(len(df) for df in dfs)

    epochs = dfs[0]["epoch"].values[:min_len]

    train = np.vstack([df["train"].values[:min_len] for df in dfs])
    test = np.vstack([df["test"].values[:min_len] for df in dfs])

    return (
        epochs,
        train.mean(axis=0),
        train.std(axis=0),
        test.mean(axis=0),
        test.std(axis=0),
    )


def plot_group(files, title, outfile):

    epochs, train_mean, train_std, test_mean, test_std = compute_mean_std(files)

    plt.figure(figsize=(8,6))

    # Train
    plt.plot(
        epochs,
        train_mean,
        linewidth=2,
        label="Train",
    )

    plt.fill_between(
        epochs,
        train_mean-train_std,
        train_mean+train_std,
        alpha=0.25,
    )

    # Test
    plt.plot(
        epochs,
        test_mean,
        linewidth=2,
        label="Test",
    )

    plt.fill_between(
        epochs,
        test_mean-test_std,
        test_mean+test_std,
        alpha=0.25,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)

    print("Saved:", outfile)
    plt.show()

def plot_compare_groups(group_specs, title, outfile, plot_train=True, plot_test=True):
    plt.figure(figsize=(8, 6))

    for label, files in group_specs:
        epochs, train_mean, train_std, test_mean, test_std = compute_mean_std(files)

        if plot_train:
            plt.plot(
                epochs,
                train_mean,
                linewidth=2,
                linestyle="--",
                label=f"{label} Train",
            )
            plt.fill_between(
                epochs,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.15,
            )

        if plot_test:
            plt.plot(
                epochs,
                test_mean,
                linewidth=2,
                label=f"{label} Test",
            )
            plt.fill_between(
                epochs,
                test_mean - test_std,
                test_mean + test_std,
                alpha=0.20,
            )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print("Saved:", outfile)
    plt.show()

# ----------------------------------------
# Figure 5: PCC - 8 layers +150 channels 
# ----------------------------------------
# plot_group(
#     pcc8l150_files,
#     "PCC Normal",
#     "pcc_Normal.png",
# )

# ----------------------------------------
# Figure 6: DC - Bottleneck layers +150 channels 
# ----------------------------------------
# plot_group(
#     bottleneck_files,
#     "DC Bottleneck layers (+150 channels)",
#     "dcbottleneck150_mean_std.png",
# )

# ----------------------------------------
# Figure 7: Compare Figure 5 and Figure 6
# ----------------------------------------

plot_compare_groups(
    [
        # ("DC Baseline 110-130", dc_baseline_110),
        ("DCT Transformer LR Reduced", dct_transformer_logs),
        ("DCT CNN + Transformer scale 0.25", dct_scale_025),
    ],
    "DC-T Transformer Reduced LR - Baseline Channels, 110-130 Hz",
    "dct_reducedlr.png",
)