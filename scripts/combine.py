import numpy as np

data = np.load("combined_data_updated.npz")
# clean_dict = {}

# for k in data.files:
#     # keep only valid channels + metadata
#     if k.startswith("H1:") and len(k) > 3:
#         clean_dict[k] = data[k]
#     elif k in ["t0", "fs", "sample_rate"]:
#         clean_dict[k] = data[k]
#     else:
#         print("Removing bad key:", repr(k))

# np.savez("combined_data_updated.npz", **clean_dict)

# print(data.files)
# ---- paths ----
combined_path = "combined_data_updated.npz"

# your new channel files (edit these paths)
new_channels = {
    "H1:ISI-GND_BRS_ETMY_RX_OUT_DQ":"H1_ISI_GND_BRS_ETMY_RX_OUT_DQ.npz",
}

output_path = "/storage/home/hcoda1/3/statachar3/r-pli77-0/deepcleanv3/data/combined_data_updated.npz"

# ---- load existing combined data ----
combined = np.load(combined_path)
data_dict = {k: combined[k] for k in combined.files}

# ---- add new channels ----
for chan_name, file_path in new_channels.items():
    new_data = np.load(file_path)

    # assume each file has a single array OR same key name
    if len(new_data.files) == 1:
        arr = new_data[new_data.files[0]]
    else:
        # if key matches channel name
        arr = new_data[chan_name]

    # sanity check: length match
    if len(arr) != len(next(iter(data_dict.values()))):
        raise ValueError(f"Length mismatch for {chan_name}")

    data_dict[chan_name] = arr
    print(f"Added {chan_name}, shape={arr.shape}")

# ---- save updated file ----
np.savez(output_path, **data_dict)

print(f"\nSaved updated dataset to: {output_path}")