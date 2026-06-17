#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import math
import pandas as pd

TARGET = "H1:GDS-CALIB_STRAIN"
CHANNEL_RE = re.compile(r"^H1:[A-Za-z0-9_-]+$")


def read_channels(path: Path):
    channels = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        channels.append(line)
    return channels


def normalize_channel(ch: str):
    """
    Converts between formats like:
      H1:HPI-HAM1_BLND_L4C_VP_IN1_DQ
      H1_HPI_HAM1_BLND_L4C_VP_IN1_DQ

    This is only for matching against the coherence CSV.
    """
    return ch.replace(":", "_").replace("-", "_")


def load_coherence(path: Path):
    df = pd.read_csv(path)

    required_cols = {"channel", "band_mean_coherence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Coherence file is missing columns: {missing}")

    df["channel_norm"] = df["channel"].astype(str).map(normalize_channel)

    coherence = {}
    for _, row in df.iterrows():
        coherence[row["channel_norm"]] = row["band_mean_coherence"]

    return coherence


def is_nan_value(x):
    try:
        return pd.isna(x) or math.isnan(float(x))
    except Exception:
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "channel_dir",
        type=Path,
        help="Directory containing plain-text channel set files",
    )
    parser.add_argument(
        "--coherence-csv",
        type=Path,
        required=True,
        help="CSV file containing channel and band_mean_coherence columns",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Expected total number of channels including target",
    )
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="Require H1:GDS-CALIB_STRAIN to be included",
    )
    parser.add_argument(
        "--allow-missing-coherence",
        action="store_true",
        help="Do not fail if a selected channel is missing from the coherence CSV",
    )

    args = parser.parse_args()

    files = sorted(args.channel_dir.glob("*.ini"))
    if not files:
        raise SystemExit(f"No .txt files found in {args.channel_dir}")

    coherence = load_coherence(args.coherence_csv)

    any_failed = False

    for path in files:
        channels = read_channels(path)
        unique_channels = list(dict.fromkeys(channels))

        duplicates = sorted({ch for ch in channels if channels.count(ch) > 1})
        bad_format = [ch for ch in channels if not CHANNEL_RE.match(ch)]
        has_target = TARGET in channels

        missing_coherence = []
        nan_coherence = []

        for ch in unique_channels:
            if ch == TARGET:
                continue

            ch_norm = normalize_channel(ch)

            if ch_norm not in coherence:
                missing_coherence.append(ch)
                continue

            val = coherence[ch_norm]
            if is_nan_value(val):
                nan_coherence.append(ch)

        failed = False

        if duplicates:
            failed = True

        if bad_format:
            failed = True

        if args.expected is not None and len(unique_channels) != args.expected:
            failed = True

        if args.require_target and not has_target:
            failed = True

        if nan_coherence:
            failed = True

        if missing_coherence and not args.allow_missing_coherence:
            failed = True

        status = "PASS" if not failed else "FAIL"
        print(f"\n[{status}] {path.name}")
        print(f"  total lines:          {len(channels)}")
        print(f"  unique channels:      {len(unique_channels)}")
        print(f"  contains target:      {has_target}")

        if args.expected is not None:
            print(f"  expected channels:    {args.expected}")

        print(f"  missing coherence:    {len(missing_coherence)}")
        print(f"  NaN band coherence:   {len(nan_coherence)}")

        if duplicates:
            print("  duplicates:")
            for ch in duplicates:
                print(f"    {ch}")

        if bad_format:
            print("  bad format:")
            for ch in bad_format[:20]:
                print(f"    {ch}")
            if len(bad_format) > 20:
                print(f"    ... and {len(bad_format) - 20} more")

        if missing_coherence:
            print("  missing from coherence CSV:")
            for ch in missing_coherence[:20]:
                print(f"    {ch}")
            if len(missing_coherence) > 20:
                print(f"    ... and {len(missing_coherence) - 20} more")

        if nan_coherence:
            print("  channels with NaN band_mean_coherence:")
            for ch in nan_coherence:
                print(f"    {ch}")

        if failed:
            any_failed = True

    if any_failed:
        raise SystemExit("\nSome channel sets failed validation.")
    else:
        print("\nAll channel sets passed validation.")


if __name__ == "__main__":
    main()