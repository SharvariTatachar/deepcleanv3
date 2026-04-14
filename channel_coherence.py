# Calculates coherence between each channel in channels.ini and the target channel (strain)

import numpy as np 
from scipy.signal import coherence 

channels = "channels.ini"
data_path = "/storage/home/hcoda1/3/statachar3/deepcleanv3/data/combined_data_updated.npz"
fs = 2048                     # sampling rate
fmin, fmax = 118, 124         # frequency band of interest
nperseg = 1024                # Welch segment length
target_channel = None         


def load_channels(ini_path):
    channels = []
    with open(ini_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            channels.append(line)
    return channels

chan_lst = load_channels(channels)
target_channel = chan_lst[0]
witness_channels = [ch for ch in chan_lst if ch != target_channel]

data = np.load(data_path)

if target_channel not in data.files:
    raise KeyError(f"Target channel {target_channel} not found in {data_path}")

strain = data[target_channel]

results = []

for ch in witness_channels:
    if ch not in data.files:
        print(f"Skipping {ch}: not found in {data_path}")
        continue

    witness = data[ch]

    if len(witness) != len(strain):
        print(f"Skipping {ch}: length mismatch ({len(witness)} vs {len(strain)})")
        continue

    freqs, coh = coherence(
        strain,
        witness,
        fs=fs,
        nperseg=nperseg
    )

    band_mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(band_mask):
        mean_band_coh = np.nan
    else:
        mean_band_coh = np.mean(coh[band_mask])

    results.append((ch, mean_band_coh))

# -------- sort and print --------
results.sort(key=lambda x: x[1], reverse=True)

print(f"\nMean coherence with {target_channel} in {fmin}-{fmax} Hz:\n")
for ch, val in results:
    print(f"{ch:60s}  {val:.6f}")
