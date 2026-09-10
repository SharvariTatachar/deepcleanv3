from pathlib import Path
import warnings

<<<<<<< HEAD:channel_coherence.py
import numpy as np
import pandas as pd
from scipy.signal import coherence
=======
import numpy as np 
from scipy.signal import coherence 

channels = "irrchans.ini"
data_path = "data/combined_data_updated.npz"

fs = 2048                     # sampling rate
fmin, fmax = 118, 124         # frequency band of interest
nperseg = 1024                # Welch segment length
target_channel = None         
>>>>>>> 35a5e30607919fdd4a607b3fe50e782148c56d3d:scripts/channel_coherence.py


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

<<<<<<< HEAD:channel_coherence.py
data_dir = Path("data/dataset")
target_file = data_dir / "H1_GDS_CALIB_STRAIN.npz"
target_key = "ts"

fs = 2048
fmin, fmax = 110, 130
nperseg = 1024
out_csv = Path("data/coherence_outputs/all_witness_coherence_110_130Hz.csv")

=======
chan_lst = load_channels(channels)
print('done loading channels')
target_channel = chan_lst[0]
witness_channels = [ch for ch in chan_lst if ch != target_channel]

data = np.load(data_path)
print('done loading data')
if target_channel not in data.files:
    raise KeyError(f"Target channel {target_channel} not found in {data_path}")
>>>>>>> 35a5e30607919fdd4a607b3fe50e782148c56d3d:scripts/channel_coherence.py

def load_timeseries(npz_path: Path, key: str = "ts") -> np.ndarray:
    """Load one 1D time series from an NPZ file."""
    with np.load(npz_path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(
                f"Key {key!r} not found in {npz_path}. "
                f"Available keys: {data.files}"
            )

        values = np.asarray(data[key], dtype=np.float64).squeeze()

    if values.ndim != 1:
        raise ValueError(
            f"Expected a 1D time series in {npz_path}, "
            f"but got shape {values.shape}"
        )

    return values

<<<<<<< HEAD:channel_coherence.py

# --------------------------------------------------
# Load target strain
# --------------------------------------------------

if not target_file.exists():
    raise FileNotFoundError(f"Target file not found: {target_file}")

strain = load_timeseries(target_file, target_key)

print(f"Target file: {target_file}")
print(f"Target samples: {len(strain)}")


# --------------------------------------------------
# Find all witness files
# --------------------------------------------------

witness_files = sorted(
    path
    for path in data_dir.glob("*.npz")
    if path.resolve() != target_file.resolve()
)

print(f"Witness files found: {len(witness_files)}")

rows = []


# --------------------------------------------------
# Compute coherence
# --------------------------------------------------

for index, witness_file in enumerate(witness_files, start=1):
    channel_name = witness_file.stem

    row = {
        "channel": channel_name,
        "filename": witness_file.name,
        "status": "ok",
        "n_samples": np.nan,
        "std": np.nan,
        "rms": np.nan,
        "time_domain_corr": np.nan,
        "band_mean_coherence": np.nan,
        "band_median_coherence": np.nan,
        "band_max_coherence": np.nan,
        "band_valid_bins": 0,
        "band_nonfinite_bins": 0,
        "band_nan_fraction": np.nan,
    }

    try:
        witness = load_timeseries(witness_file, key="ts")
        row["n_samples"] = len(witness)

        if len(witness) != len(strain):
            row["status"] = (
                f"length_mismatch_{len(witness)}_vs_{len(strain)}"
            )
            rows.append(row)
            continue

        finite_mask = np.isfinite(strain) & np.isfinite(witness)

        if finite_mask.sum() < nperseg:
            row["status"] = "too_few_finite_samples"
            rows.append(row)
            continue

        strain_clean = strain[finite_mask]
        witness_clean = witness[finite_mask]

        if not np.all(finite_mask):
            row["status"] = "contains_nonfinite_samples"

        witness_std = float(np.std(witness_clean))
        witness_rms = float(np.sqrt(np.mean(witness_clean**2)))

        row["std"] = witness_std
        row["rms"] = witness_rms

        if witness_std <= 1e-20:
            row["status"] = "constant_or_zero_channel"
            rows.append(row)
            continue

        strain_std = float(np.std(strain_clean))

        if strain_std > 0:
            row["time_domain_corr"] = float(
                np.corrcoef(strain_clean, witness_clean)[0, 1]
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            freqs, coh = coherence(
                strain_clean,
                witness_clean,
                fs=fs,
                nperseg=min(nperseg, len(strain_clean)),
            )

        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_coh = np.asarray(coh[band_mask], dtype=np.float64)

        if band_coh.size == 0:
            row["status"] = "no_frequency_bins_in_band"
            rows.append(row)
            continue

        band_coh[~np.isfinite(band_coh)] = np.nan

        valid_mask = np.isfinite(band_coh)
        n_valid = int(valid_mask.sum())
        n_nonfinite = int((~valid_mask).sum())

        row["band_valid_bins"] = n_valid
        row["band_nonfinite_bins"] = n_nonfinite
        row["band_nan_fraction"] = n_nonfinite / len(band_coh)

        if n_valid == 0:
            row["status"] = "no_finite_coherence_bins"
            rows.append(row)
            continue

        row["band_mean_coherence"] = float(np.nanmean(band_coh))
        row["band_median_coherence"] = float(np.nanmedian(band_coh))
        row["band_max_coherence"] = float(np.nanmax(band_coh))

        if n_nonfinite > 0:
            row["status"] = "partial_nonfinite_coherence"

    except Exception as exc:
        row["status"] = f"error: {type(exc).__name__}: {exc}"

    rows.append(row)

    if index % 50 == 0 or index == len(witness_files):
        print(f"Processed {index}/{len(witness_files)} files")


# --------------------------------------------------
# Save results
# --------------------------------------------------

df = pd.DataFrame(rows)

df["_valid"] = np.isfinite(df["band_mean_coherence"])

df = (
    df.sort_values(
        by=["_valid", "band_mean_coherence"],
        ascending=[False, False],
        na_position="last",
=======
    if len(witness) != len(strain):
        print(f"Skipping {ch}: length mismatch ({len(witness)} vs {len(strain)})")
        continue
    print('wit: ', ch)
    freqs, coh = coherence(
        strain,
        witness,
        fs=fs,
        nperseg=nperseg
>>>>>>> 35a5e30607919fdd4a607b3fe50e782148c56d3d:scripts/channel_coherence.py
    )
    .drop(columns="_valid")
    .reset_index(drop=True)
)

# Rank valid channels only
df.insert(0, "rank", pd.Series(dtype="Int64"))

valid_mask = np.isfinite(df["band_mean_coherence"])
df.loc[valid_mask, "rank"] = np.arange(1, valid_mask.sum() + 1)

out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print(f"\nSaved {len(df)} witness results to:")
print(out_csv.resolve())

print("\nTop 20 channels:")
print(
    df.loc[
        valid_mask,
        [
            "rank",
            "channel",
            "band_mean_coherence",
            "band_median_coherence",
            "band_valid_bins",
            "status",
        ],
    ]
    .head(20)
    .to_string(index=False)
)

print("\nStatus counts:")
print(df["status"].value_counts(dropna=False).to_string())