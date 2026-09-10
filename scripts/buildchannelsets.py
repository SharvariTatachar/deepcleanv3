#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_CHANNEL = "H1:GDS-CALIB_STRAIN"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build granular witness-channel sets from coherence CSV output."
    )

    parser.add_argument(
        "--coherence-csv",
        type=Path,
        required=True,
        help="CSV produced by channel_coherence.py.",
    )
    parser.add_argument(
        "--baseline-ini",
        type=Path,
        required=True,
        help="Baseline channel list. Target must be the first channel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("channelconfigs/coherence_granular"),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100, 150],
        help="Numbers of additional witness channels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for mixed channel sets.",
    )
    parser.add_argument(
        "--coherence-column",
        type=str,
        default="band_mean_coherence",
        help="Coherence column to rank by.",
    )
    parser.add_argument(
        "--channel-column",
        type=str,
        default="channel",
    )
    parser.add_argument(
        "--status-column",
        type=str,
        default="status",
    )
    parser.add_argument(
        "--require-status-ok",
        action="store_true",
        help="Keep only rows whose status is 'ok'.",
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        help="Optional directory of per-channel NPZ files. "
             "When provided, only channels with matching NPZ files are retained.",
    )

    return parser.parse_args()


def load_channel_list(path: Path) -> list[str]:
    channels = []

    with path.open() as file:
        for line in file:
            line = line.strip()

            if line and not line.startswith("#"):
                channels.append(line)

    if not channels:
        raise ValueError(f"No channels found in {path}")

    return channels


def channel_to_npz_name(channel: str) -> str:
    """
    Convert:
        H1:PEM-CS_ACC_PSL_TABLE1_Z_DQ
    to:
        H1_PEM_CS_ACC_PSL_TABLE1_Z_DQ.npz
    """
    return channel.replace(":", "_").replace("-", "_") + ".npz"


def load_ranked_candidates(
    csv_path: Path,
    baseline_channels: list[str],
    channel_column: str,
    coherence_column: str,
    status_column: str,
    require_status_ok: bool,
    npz_dir: Path | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {channel_column, coherence_column}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"CSV is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    df[channel_column] = df[channel_column].astype(str).str.strip()
    df[coherence_column] = pd.to_numeric(
        df[coherence_column],
        errors="coerce",
    )

    # Drop invalid coherence values, including inf and -inf.
    valid = np.isfinite(df[coherence_column].to_numpy())
    df = df.loc[valid].copy()

    if require_status_ok and status_column in df.columns:
        df = df.loc[df[status_column].astype(str).str.lower() == "ok"].copy()

    excluded = set(baseline_channels)
    excluded.add(TARGET_CHANNEL)

    df = df.loc[~df[channel_column].isin(excluded)].copy()

    # Remove repeated channel rows.
    df = df.drop_duplicates(subset=channel_column, keep="first")

    if npz_dir is not None:
        exists = df[channel_column].map(
            lambda channel: (
                npz_dir / channel_to_npz_name(channel)
            ).exists()
        )

        missing_count = int((~exists).sum())

        if missing_count:
            print(
                f"Excluding {missing_count} channels without matching NPZ files."
            )

        df = df.loc[exists].copy()

    return df.sort_values(
        coherence_column,
        ascending=True,
    ).reset_index(drop=True)


def select_low(df: pd.DataFrame, size: int) -> pd.DataFrame:
    return df.head(size).copy()


def select_high(df: pd.DataFrame, size: int) -> pd.DataFrame:
    return df.tail(size).sort_values(
        df.columns[df.columns != "channel"][0]
    ).copy()


def select_mixed(
    df: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select channels across the entire coherence distribution.

    Stratified sampling is used instead of unrestricted random sampling so
    low-, middle-, and high-coherence regions are represented.
    """
    if size >= len(df):
        return df.copy()

    rng = np.random.default_rng(seed)

    # Split ranked candidates into approximately equal strata.
    n_strata = min(10, size, len(df))
    strata = np.array_split(np.arange(len(df)), n_strata)

    base_per_stratum = size // n_strata
    remainder = size % n_strata

    selected_indices = []

    for stratum_index, indices in enumerate(strata):
        count = base_per_stratum + int(stratum_index < remainder)
        count = min(count, len(indices))

        if count:
            selected_indices.extend(
                rng.choice(indices, size=count, replace=False).tolist()
            )

    selected_indices = sorted(selected_indices)

    return df.iloc[selected_indices].copy()


def write_ini(
    output_path: Path,
    baseline_channels: list[str],
    added_channels: list[str],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve target as the first entry.
    ordered = []

    for channel in baseline_channels + added_channels:
        if channel not in ordered:
            ordered.append(channel)

    with output_path.open("w") as file:
        for channel in ordered:
            file.write(f"{channel}\n")


def main():
    args = parse_args()

    baseline_channels = load_channel_list(args.baseline_ini)

    if baseline_channels[0] != TARGET_CHANNEL:
        raise ValueError(
            f"Expected target channel first in {args.baseline_ini}, but found "
            f"{baseline_channels[0]!r}"
        )

    candidates = load_ranked_candidates(
        csv_path=args.coherence_csv,
        baseline_channels=baseline_channels,
        channel_column=args.channel_column,
        coherence_column=args.coherence_column,
        status_column=args.status_column,
        require_status_ok=args.require_status_ok,
        npz_dir=args.npz_dir,
    )

    if candidates.empty:
        raise RuntimeError("No valid non-baseline candidate channels remain.")

    print(f"Baseline total, including target: {len(baseline_channels)}")
    print(f"Candidate witnesses available: {len(candidates)}")
    print(
        "Coherence range: "
        f"{candidates[args.coherence_column].min():.6g} to "
        f"{candidates[args.coherence_column].max():.6g}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    selectors = {
        "low": lambda frame, size: select_low(frame, size),
        "high": lambda frame, size: select_high(frame, size),
        "mixed": lambda frame, size: select_mixed(
            frame,
            size,
            seed=args.seed + size,
        ),
    }

    for size in args.sizes:
        if size > len(candidates):
            print(
                f"Skipping size {size}: only {len(candidates)} candidates exist."
            )
            continue

        for group_name, selector in selectors.items():
            selected = selector(candidates, size)

            added_channels = selected[args.channel_column].tolist()

            set_name = f"add{size}{group_name}"
            ini_path = args.output_dir / f"{set_name}.ini"
            csv_path = args.output_dir / f"{set_name}_selected.csv"
            json_path = args.output_dir / f"{set_name}.json"

            write_ini(
                ini_path,
                baseline_channels=baseline_channels,
                added_channels=added_channels,
            )

            selected_output = selected[
                [args.channel_column, args.coherence_column]
            ].copy()

            selected_output.insert(
                0,
                "selection_rank",
                np.arange(1, len(selected_output) + 1),
            )
            selected_output.to_csv(csv_path, index=False)

            metadata = {
                "name": set_name,
                "group": group_name,
                "requested_added_channels": size,
                "actual_added_channels": len(added_channels),
                "baseline_total_including_target": len(baseline_channels),
                "baseline_witnesses": len(baseline_channels) - 1,
                "total_channels_including_target": (
                    len(baseline_channels) + len(added_channels)
                ),
                "total_witnesses": (
                    len(baseline_channels) - 1 + len(added_channels)
                ),
                "target_channel": TARGET_CHANNEL,
                "coherence_csv": str(args.coherence_csv),
                "coherence_column": args.coherence_column,
                "seed": args.seed if group_name == "mixed" else None,
                "mean_added_coherence": float(
                    selected[args.coherence_column].mean()
                ),
                "min_added_coherence": float(
                    selected[args.coherence_column].min()
                ),
                "max_added_coherence": float(
                    selected[args.coherence_column].max()
                ),
                "channels": baseline_channels + added_channels,
                "added_channels": added_channels,
            }

            with json_path.open("w") as file:
                json.dump(metadata, file, indent=2)

            metadata_rows.append(
                {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"channels", "added_channels"}
                }
            )

            print(
                f"Created {set_name}: "
                f"{len(added_channels)} added, "
                f"{len(baseline_channels) + len(added_channels)} total "
                f"including target"
            )

    summary = pd.DataFrame(metadata_rows)
    summary_path = args.output_dir / "channel_set_summary.csv"
    summary.to_csv(summary_path, index=False)

    candidates.to_csv(
        args.output_dir / "ranked_candidate_channels.csv",
        index=False,
    )

    print(f"\nSaved channel sets to: {args.output_dir.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()