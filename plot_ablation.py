from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BAND = "110"
BAND_LABEL = "110 - 130 Hz"
MODELS = ("pcc", "dct", "dc")
SEEDS = [0]
# SEEDS = (0, 1, 2)
MAX_EPOCH_ZOOM = 20
OUTPUT_DIR = Path(".")
MODEL_COLORS = {"pcc": "tab:blue", "dct": "tab:orange", "dc": "tab:green"}


def get_run_dir(model, seed):
    root = Path(f"runs_{model}_baseline_{BAND}")
    candidates = [
        root / f"baseline_seed{seed}",
        root / f"{model}_seed{seed}",
        root / f"seed{seed}",
        root.parent / f"{root.name}_seed{seed}",
    ]
    for candidate in candidates:
        if (candidate / "metrics" / "loss.dat").is_file():
            return candidate
    print(f"Missing {model.upper()} seed {seed}; checked: " + ", ".join(map(str, candidates)))
    return None


def load_loss_curve(run_dir):
    path = run_dir / "metrics" / "loss.dat"
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (ValueError, OSError) as exc:
        print(f"Cannot read {path}: {exc}")
        return None
    if data.shape[1] < 4:
        print(f"Expected at least 4 columns in {path}, got {data.shape[1]}")
        return None
    epoch, train, val = data[:, 1], data[:, -2], data[:, -1]
    valid = np.isfinite(epoch) & np.isfinite(train) & np.isfinite(val)
    epoch, train, val = epoch[valid], train[valid], val[valid]
    if len(epoch) < 2:
        print(f"Not enough valid epochs in {path}")
        return None
    order = np.argsort(epoch, kind="stable")
    epoch, train, val = epoch[order], train[order], val[order]
    unique_epoch, indices = np.unique(epoch, return_index=True)
    if len(unique_epoch) < 2:
        print(f"Not enough distinct epochs in {path}")
        return None
    return {"epoch": unique_epoch, "train": train[indices], "val": val[indices]}


def summarize_curves(curves, max_epoch=None, points_per_epoch=10):
    start = max(curve["epoch"][0] for curve in curves)
    end = min(curve["epoch"][-1] for curve in curves)
    if max_epoch is not None:
        end = min(end, max_epoch)
    if end <= start:
        return None
    epochs = np.linspace(start, end, max(2, int((end - start) * points_per_epoch) + 1))
    vals = np.array([np.interp(epochs, c["epoch"], c["val"]) for c in curves])
    trains = np.array([np.interp(epochs, c["epoch"], c["train"]) for c in curves])
    return epochs, {
        "val": (vals.mean(axis=0), vals.std(axis=0, ddof=0)),
        "train": (trains.mean(axis=0), trains.std(axis=0, ddof=0)),
    }


def plot_comparison(grouped_curves, filename, max_epoch=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in MODELS:
        curves = grouped_curves.get(model, [])
        if not curves:
            continue
        summary = summarize_curves(curves, max_epoch=max_epoch)
        if summary is None:
            print(f"Cannot plot {model.upper()}: no shared epoch interval")
            continue
        epochs, losses = summary
        for loss_type, linestyle in (("val", "-"), ("train", "--")):
            mean, std = losses[loss_type]
            line, = ax.plot(
                epochs, mean, linewidth=2, linestyle=linestyle,
                color=MODEL_COLORS[model],
                label=f"{model.upper()} {loss_type.capitalize()} (n={len(curves)})",
            )
            if len(curves) > 1:
                ax.fill_between(epochs, mean - std, mean + std,
                                color=line.get_color(), alpha=0.12)
    if not ax.lines:
        plt.close(fig)
        raise RuntimeError("No valid model curves available to plot")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    title = f"PCC vs DCT vs DC — {BAND_LABEL}"
    if max_epoch is not None:
        title += f" (through epoch {max_epoch})"
        ax.set_xlim(0, max_epoch)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


def main():
    grouped_curves = {}
    for model in MODELS:
        curves = []
        best_losses = []
        for seed in SEEDS:
            run_dir = get_run_dir(model, seed)
            if run_dir is None:
                continue
            curve = load_loss_curve(run_dir)
            if curve is None:
                continue
            curves.append(curve)
            best = float(np.min(curve["val"]))
            best_losses.append(best)
            print(f"{model.upper()} seed {seed}: best val={best:.6f} ({run_dir})")
        if not curves:
            print(f"No valid runs found for {model.upper()}")
            continue
        if len(curves) != len(SEEDS):
            print(f"WARNING: {model.upper()} has {len(curves)}/{len(SEEDS)} seeds")
        grouped_curves[model] = curves
        print(f"{model.upper()}: mean best val={np.mean(best_losses):.6f}, "
              f"std={np.std(best_losses, ddof=0):.6f}, n={len(best_losses)}\n")
    plot_comparison(grouped_curves, f"model_comp_{BAND}.png")
    plot_comparison(grouped_curves, f"model_comp_epoch_{BAND}.png", MAX_EPOCH_ZOOM)


if __name__ == "__main__":
    main()

