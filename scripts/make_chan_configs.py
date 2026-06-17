"""
script instructions
python3 make_chan_configs.py \
  --coherence-csv /storage/home/hcoda1/3/statachar3/r-pli77-0/deepcleanv3/data/coherence_outputs/coherence_summary.csv \
  --baseline-channels ogchannels.ini \
  --out-dir channel_sets \
  --added-k 20 \
  --seed 0
"""


import csv
import json
import argparse
from pathlib import Path
import numpy as np

def read_channel_list(path):
    channels = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                continue

            if line.startswith("#"):
                continue

            channels.append(line)

    print(f"Loaded {len(channels)} baseline channels from {path}")

    return channels

def read_coherence_csv(path):
    import csv
    import numpy as np

    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        print("CSV columns:")
        print(reader.fieldnames)

        for r in reader:

            # ----- coherence -----
            raw_coh = r.get("band_mean_coherence", "").strip()

            if raw_coh.lower() in ("", "nan", "none"):
                coh = np.nan
            else:
                try:
                    coh = float(raw_coh)
                except ValueError:
                    coh = np.nan

            # ----- nan fraction -----
            raw_nan = r.get("band_nan_fraction", "").strip()

            if raw_nan.lower() in ("", "nan", "none"):
                nan_frac = np.nan
            else:
                try:
                    nan_frac = float(raw_nan)
                except ValueError:
                    nan_frac = np.nan

            rows.append({
                "channel": r["channel"],
                "coherence": coh,
                "band_nan_fraction": nan_frac,
            })

    print(f"Rows read from CSV: {len(rows)}")

    # keep only rows with finite coherence
    valid_rows = [
        r for r in rows
        if np.isfinite(r["coherence"])
    ]

    print(f"Rows with finite coherence: {len(valid_rows)}")

    if len(valid_rows) > 0:
        cohs = [r["coherence"] for r in valid_rows]

        print(f"Min coherence: {np.min(cohs):.6f}")
        print(f"Median coherence: {np.median(cohs):.6f}")
        print(f"Max coherence: {np.max(cohs):.6f}")

    return valid_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coherence-csv", required=True)
    parser.add_argument("--out-dir", default="channel_sets")
    parser.add_argument("--target", default="H1_GDS_CALIB_STRAIN")
    parser.add_argument("--baseline-k", type=int, default=35)
    parser.add_argument("--added-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-channels", required=True)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    rows = read_coherence_csv(args.coherence_csv)

    print(f"Valid coherence rows: {len(rows)}")

    baseline_channels = read_channel_list(args.baseline_channels)
    baseline_channels = [ch for ch in baseline_channels if ch != args.target]
    baseline_set = set(baseline_channels)

    candidate_rows = [
        r for r in rows
        if r["channel"] not in baseline_set
        and r["channel"] != args.target
    ]

    print(f"Baseline channels: {len(baseline_channels)}")
    print(f"Candidate added channels: {len(candidate_rows)}")

    if len(candidate_rows) == 0:
        raise ValueError(
            "No candidate added channels found. This means either the coherence CSV "
            "has no valid coherence values, or all valid rows are in the baseline."
        )

    cohs = np.array([r["coherence"] for r in candidate_rows])
    q25 = np.percentile(cohs, 25)
    q50 = np.percentile(cohs, 50)
    q75 = np.percentile(cohs, 75)

    print(f"Q25 = {q25:.6f}")
    print(f"Q50 = {q50:.6f}")
    print(f"Q75 = {q75:.6f}")

    # Define coherence bins for added channels.
    bins = [
    ("high_added", q75, np.inf),
    ("medium_added", q50, q75),
    ("low_added", q25, q50),
    ("very_low_added", -np.inf, q25),
    ]
    manifest = []

    for bin_name, lo, hi in bins:
        bin_rows = [
            r for r in candidate_rows
            if lo <= r["coherence"] < hi
        ]

        if len(bin_rows) == 0:
            print(f"Skipping {bin_name}: no channels found")
            continue

        if len(bin_rows) < args.added_k:
            print(
                f"Warning: {bin_name} only has {len(bin_rows)} channels; "
                f"using all of them."
            )
            selected_rows = bin_rows
        else:
            selected_rows = list(rng.choice(bin_rows, size=args.added_k, replace=False))

        added_channels = [r["channel"] for r in selected_rows]
        added_coherences = [r["coherence"] for r in selected_rows]

        all_channels = baseline_channels + added_channels

        config = {
            "name": bin_name,
            "target": args.target,
            "baseline_k": args.baseline_k,
            "added_k": len(added_channels),
            "avg_added_coherence": float(np.mean(added_coherences)),
            "coherence_bin": {
                "lower": float(lo),
                "upper": float(hi) if np.isfinite(hi) else None
            },
            "min_added_coherence": float(np.min(added_coherences)),
            "max_added_coherence": float(np.max(added_coherences)),
            "channels": all_channels,
            "baseline_channels": baseline_channels,
            "added_channels": added_channels,
            "added_channel_coherences": dict(zip(added_channels, added_coherences)),
        }

        out_path = out_dir / f"{bin_name}_{args.added_k}seed{args.seed}.json"

        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

        manifest.append({
            "name": bin_name,
            "path": str(out_path),
            "avg_added_coherence": config["avg_added_coherence"],
            "added_k": config["added_k"],
            "total_channels": len(all_channels),
        })

        print(
            f"{bin_name}: avg coherence={config['avg_added_coherence']:.5f}, "
            f"channels={len(all_channels)}, saved={out_path}"
        )

    with open(out_dir / f"manifest_{args.added_k}seed{args.seed}.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()