from pathlib import Path
import warnings
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence

data_dir = Path("/storage/home/hcoda1/3/statachar3/r-pli77-0/deepcleanv3/data/dataset")
out_dir = Path("coherence_outputs")
out_dir.mkdir(exist_ok=True)

target_name = "H1_GDS_CALIB_STRAIN"
target_file = data_dir / f"{target_name}.npz"

fs = 2048
fmin, fmax = 110, 130
nperseg = 4096
noverlap = nperseg // 2

# strain = np.load(target_file)["ts"].squeeze().astype(np.float32)
strain = np.load(target_file)["ts"].squeeze().astype(np.float64)

rows = []
all_coh = []
all_names = []
freqs_ref = None

npz_files = sorted(data_dir.glob("*.npz"))

for i, path in enumerate(npz_files):
    ch = path.stem

    if ch == target_name:
        continue

    if i % 25 == 0:
        print(f"[{i}/{len(npz_files)}] {ch}", flush=True)

    # x = np.load(path)["ts"].squeeze().astype(np.float32)
    x = np.load(path)["ts"].squeeze().astype(np.float64)

    n = min(len(x), len(strain))
    x = x[:n]
    y = strain[:n]

    with warnings.catch_warnings(): 
        warnings.simplefilter('ignore', RuntimeWarning)
        freqs, coh = coherence(
            x,
            y,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )

    if freqs_ref is None:
        print("freq range:", freqs[0], freqs[-1])
        print("freq step:", freqs[1] - freqs[0])
        print("band bins:", np.sum((freqs >= fmin) & (freqs <= fmax)))

    if freqs_ref is None:
        freqs_ref = freqs

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    band_vals = coh[band_mask]
    valid_band_vals = band_vals[np.isfinite(band_vals)]

    band_valid_bins = len(valid_band_vals)
    band_total_bins = len(band_vals)

    if band_valid_bins == 0: 
        band_mean = np.nan
        band_std = np.nan 
    else: 
        band_mean = float(np.mean(valid_band_vals))
        band_std = float(np.std(valid_band_vals))

    finite_coh = coh[np.isfinite(coh)]

    if len(finite_coh) == 0: 
        mean_coh = np.nan
        std_coh = np.nan
        max_coh = np.nan
        freq_at_max = np.nan
    else:
        mean_coh = float(np.mean(finite_coh))
        std_coh = float(np.std(finite_coh))
        max_idx = np.nanargmax(coh)
        max_coh = float(coh[max_idx])
        freq_at_max = float(freqs[max_idx])
    
    if np.std(x) == 0 or np.std(y) == 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(x, y)[0, 1])

    rows.append({
        "channel": ch,
        "mean_coherence_all_freqs": mean_coh,
        "std_coherence_all_freqs": std_coh,
        "max_coherence": max_coh,
        "freq_at_max_coherence": freq_at_max,
        "band_mean_coherence": band_mean,
        "band_std_coherence": band_std,
        "band_valid_bins": band_valid_bins, 
        "band_total_bins": band_total_bins, 
        "band_nan_fraction": 1- band_valid_bins/band_total_bins,
        "time_domain_corr": float(np.corrcoef(x, y)[0, 1]),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
    })

    all_coh.append(coh.astype(np.float32))
    all_names.append(ch)

rows = sorted(
    rows,
    key=lambda r: -1 if np.isnan(r["band_mean_coherence"]) else r["band_mean_coherence"],
    reverse=True,
)


with open(out_dir / "coherence_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

with open(out_dir / "top_channels_by_band_coherence.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows[:50])

all_coh = np.asarray(all_coh)

mean_coh = np.nanmean(all_coh, axis=0)
std_coh = np.nanstd(all_coh, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(freqs_ref, mean_coh, label="Mean coherence")
plt.fill_between(freqs_ref, mean_coh - std_coh, mean_coh + std_coh, alpha=0.3)
plt.axvspan(fmin, fmax, alpha=0.2, label=f"{fmin}-{fmax} Hz")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Coherence with strain")
plt.title("Mean ± Std Coherence Across Channels")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "coherence_mean_std.png", dpi=200)
plt.close()

top_names = [r["channel"] for r in rows[:40]]
top_idx = [all_names.index(ch) for ch in top_names]
top_coh = all_coh[top_idx]

plt.figure(figsize=(12, 8))
plt.imshow(
    top_coh,
    aspect="auto",
    origin="lower",
    extent=[freqs_ref[0], freqs_ref[-1], 0, len(top_names)],
)
plt.colorbar(label="Coherence")
plt.yticks(np.arange(len(top_names)) + 0.5, top_names, fontsize=6)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Channel")
plt.title("Top Channels by Band-Averaged Coherence")
plt.tight_layout()
plt.savefig(out_dir / "coherence_heatmap_top40.png", dpi=200)
plt.close()

print("Done.")
print(f"Saved outputs to {out_dir}")