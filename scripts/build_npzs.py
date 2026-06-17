from pathlib import Path
import zipfile
import numpy as np
from numpy.lib.format import write_array

DATASET_DIR = Path("data/dataset")
OUT_DIR = Path("data/channel_ablation_npzs")
CHANNELCONFIG_DIR = Path("channelconfigs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETS = {
    "add400high": CHANNELCONFIG_DIR/"add400highest.ini",
    "add400low": CHANNELCONFIG_DIR/"add400lowest.ini",
    "add400mixed": CHANNELCONFIG_DIR/"add400mixed.ini",
}

TARGET = "H1:GDS-CALIB_STRAIN"


def canonical(ch):
    return ch.replace(":", "_").replace("-", "_").upper()


def channel_to_file(ch):
    return DATASET_DIR / f"{ch.replace(':', '_').replace('-', '_')}.npz"


def load_channels(path):
    channels = []
    seen = set()

    with open(path) as f:
        for line in f:
            ch = line.strip().strip('"')
            if not ch or not ch.startswith("H1:"):
                continue

            key = canonical(ch)
            if key in seen:
                continue

            channels.append(ch)
            seen.add(key)

    if not channels or channels[0] != TARGET:
        raise ValueError(f"{path} must start with {TARGET}")

    return channels


def load_ts(ch):
    path = channel_to_file(ch)
    if not path.exists():
        raise FileNotFoundError(f"Missing {ch}: {path}")

    with np.load(path, allow_pickle=True) as f:
        if "ts" not in f:
            raise KeyError(f"{path} missing key 'ts'")
        return f["ts"]


def save_npz_streaming(out_path, channels):
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        n0 = None

        for i, ch in enumerate(channels, 1):
            arr = load_ts(ch).astype(np.float32, copy=False)

            if n0 is None:
                n0 = len(arr)
            elif len(arr) != n0:
                arr = arr[:n0]

            print(f"{i}/{len(channels)} {ch} {arr.shape}")

            with zf.open(ch + ".npy", "w") as f:
                write_array(f, arr, allow_pickle=False)

    return n0


def build_one(name, ini_path):
    channels = load_channels(ini_path)

    out_npz = OUT_DIR / f"{name}.npz"
    out_ini = OUT_DIR / f"{name}_channels.ini"

    if out_npz.exists():
        print(f"Skipping existing {out_npz}")
        return

    print(f"\nBuilding {name}")
    print(f"Unique channels: {len(channels)}")

    n = save_npz_streaming(out_npz, channels)

    with open(out_ini, "w") as f:
        for ch in channels:
            f.write(ch + "\n")

    print(f"Saved {out_npz}")
    print(f"Saved cleaned channels {out_ini}")
    print(f"Samples: {n}")
    print(f"Seconds at 2048 Hz: {n / 2048}")


def main():
    for name, ini in SETS.items():
        build_one(name, ini)


if __name__ == "__main__":
    main()