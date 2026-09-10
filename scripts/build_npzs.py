from pathlib import Path
import zipfile
<<<<<<< HEAD

import numpy as np
from numpy.lib.format import write_array


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

DATASET_DIR = Path("data/dataset")
CHANNELCONFIG_DIR = Path("channelconfigs")
OUT_DIR = Path("data/channel_ablation_npzs")

# Change these paths if your directory names differ.
CHANNEL_SETS = {
    "add200low": Path("channelconfigs/110-130/granular/add200low.ini"),
}

TARGET = "H1:GDS-CALIB_STRAIN"
SAMPLE_RATE = 2048

# False: skip combined NPZ files that already exist.
# True: rebuild and replace them.
OVERWRITE = False

OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Channel-name utilities
# --------------------------------------------------
=======
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

>>>>>>> 35a5e30607919fdd4a607b3fe50e782148c56d3d

def canonical(ch):
    return ch.replace(":", "_").replace("-", "_").upper()


def channel_to_file(ch):
<<<<<<< HEAD
    """
    Supports both original channel names:

        H1:GDS-CALIB_STRAIN

    and already-canonical filename-style names:

        H1_GDS_CALIB_STRAIN
    """
    filename = canonical(ch) + ".npz"
    return DATASET_DIR / filename
=======
    return DATASET_DIR / f"{ch.replace(':', '_').replace('-', '_')}.npz"
>>>>>>> 35a5e30607919fdd4a607b3fe50e782148c56d3d


def load_channels(path):
    channels = []
    seen = set()

    with open(path) as f:
        for line in f:
            ch = line.strip().strip('"')
<<<<<<< HEAD

            if not ch or ch.startswith("#"):
                continue

            # Accept both colon-style and filename-style channel names.
            if not (ch.startswith("H1:") or ch.startswith("H1_")):
                continue

            key = canonical(ch)

            if key in seen:
                print(f"Warning: duplicate channel ignored: {ch}")
=======
            if not ch or not ch.startswith("H1:"):
                continue

            key = canonical(ch)
            if key in seen:
>>>>>>> 35a5e30607919fdd4a607b3fe50e782148c56d3d
                continue

            channels.append(ch)
            seen.add(key)

<<<<<<< HEAD
    if not channels:
        raise ValueError(f"No channels found in {path}")

    # Compare canonically because target may be written with underscores.
    if canonical(channels[0]) != canonical(TARGET):
        raise ValueError(
            f"{path} must start with {TARGET}; "
            f"found {channels[0]}"
        )
=======
    if not channels or channels[0] != TARGET:
        raise ValueError(f"{path} must start with {TARGET}")
>>>>>>> 35a5e30607919fdd4a607b3fe50e782148c56d3d

    return channels


<<<<<<< HEAD
def load_ts(channel):
    """Load the `ts` array for one channel."""
    path = channel_to_file(channel)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing individual channel file for {channel}: {path}"
        )

    with np.load(path, allow_pickle=False) as data:
        if "ts" not in data.files:
            raise KeyError(
                f"{path} is missing the required key 'ts'. "
                f"Available keys: {data.files}"
            )

        array = np.asarray(data["ts"])

    array = np.squeeze(array)

    if array.ndim != 1:
        raise ValueError(
            f"{channel} must contain a 1D time series, "
            f"but its shape is {array.shape}"
        )

    return array


def validate_channel_files(channels):
    """
    Check that every channel file exists and contains the same number
    of samples before beginning the combined write.
    """
    expected_length = None
    errors = []

    for index, channel in enumerate(channels, start=1):
        try:
            array = load_ts(channel)

            if expected_length is None:
                expected_length = len(array)
            elif len(array) != expected_length:
                errors.append(
                    f"{channel}: length {len(array)}, "
                    f"expected {expected_length}"
                )

        except Exception as exc:
            errors.append(f"{channel}: {type(exc).__name__}: {exc}")

        if index % 50 == 0:
            print(
                f"  Validated {index}/{len(channels)} channel files"
            )

    if errors:
        error_text = "\n".join(f"  - {error}" for error in errors)

        raise RuntimeError(
            "Channel-set validation failed:\n"
            f"{error_text}"
        )

    return expected_length


# --------------------------------------------------
# Combined NPZ writing
# --------------------------------------------------

def save_npz_streaming(out_path, channels, expected_length):
    """
    Write one combined NPZ without loading all channels into memory.

    Each array is stored using its original channel name as the NPZ key.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = out_path.with_suffix(".npz.tmp")

    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as zip_file:

            for index, channel in enumerate(channels, start=1):
                array = load_ts(channel).astype(
                    np.float32,
                    copy=False,
                )

                if len(array) != expected_length:
                    raise ValueError(
                        f"{channel} changed length during writing: "
                        f"{len(array)} versus {expected_length}"
                    )

                print(
                    f"  {index}/{len(channels)} "
                    f"{channel} {array.shape}"
                )

                # np.load() will expose this key as the original
                # colon-separated channel name.
                with zip_file.open(f"{channel}.npy", "w") as file:
                    write_array(
                        file,
                        array,
                        allow_pickle=False,
                    )

        temporary_path.replace(out_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


# --------------------------------------------------
# Build one channel set
# --------------------------------------------------

def build_one(ini_path, out_npz):
    channels = load_channels(ini_path)

    out_channels = out_npz.with_name(
        f"{out_npz.stem}_channels.ini"
    )

    if out_npz.exists() and not OVERWRITE:
        print(f"\nSkipping existing file: {out_npz}")
        return "skipped"

    print("\n" + "=" * 70)
    print(f"Source INI:       {ini_path}")
    print(f"Output NPZ:       {out_npz}")
    print(f"Channels total:   {len(channels)}")
    print(f"Witness channels: {len(channels) - 1}")
    print("=" * 70)

    expected_length = validate_channel_files(channels)

    save_npz_streaming(
        out_path=out_npz,
        channels=channels,
        expected_length=expected_length,
    )

    out_channels.parent.mkdir(parents=True, exist_ok=True)

    with out_channels.open("w") as file:
        for channel in channels:
            file.write(f"{channel}\n")

    print(f"\nSaved combined data:    {out_npz}")
    print(f"Saved cleaned channels: {out_channels}")
    print(f"Samples per channel:    {expected_length}")
    print(
        f"Seconds at {SAMPLE_RATE} Hz:   "
        f"{expected_length / SAMPLE_RATE:.2f}"
    )

    return "built"


# --------------------------------------------------
# Locate all INI files
# --------------------------------------------------

def discover_channel_sets():
    """
    Build only the add200low and add400low channel sets.

    Inputs:
        channelconfigs/110-130/add200low.ini
        channelconfigs/110-130/add400low.ini

    Outputs:
        data/channel_ablation_npzs/110-130/add200low.npz
        data/channel_ablation_npzs/110-130/add400low.npz
    """
    channel_sets = []

    for set_name, ini_path in CHANNEL_SETS.items():
        if not ini_path.exists():
            print(f"Warning: channel-config file does not exist: {ini_path}")
            continue

        out_npz = OUT_DIR / "110-130" / f"{set_name}.npz"
        channel_sets.append((ini_path, out_npz))

    return channel_sets


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Individual-channel dataset directory not found: "
            f"{DATASET_DIR}"
        )

    channel_sets = discover_channel_sets()

    if not channel_sets:
        raise RuntimeError(
            "No channel-set INI files were found. Check BAND_DIRS."
        )

    print(f"\nTotal channel sets found: {len(channel_sets)}")

    built = 0
    skipped = 0
    failed = []

    for ini_path, out_npz in channel_sets:
        try:
            status = build_one(ini_path, out_npz)

            if status == "built":
                built += 1
            elif status == "skipped":
                skipped += 1

        except Exception as exc:
            failed.append(
                {
                    "ini": str(ini_path),
                    "output": str(out_npz),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            print(
                f"\nFAILED: {ini_path}\n"
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 70)
    print("BUILD SUMMARY")
    print("=" * 70)
    print(f"Built:   {built}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {len(failed)}")

    if failed:
        print("\nFailed channel sets:")

        for item in failed:
            print(f"\n  INI:   {item['ini']}")
            print(f"  NPZ:   {item['output']}")
            print(f"  Error: {item['error']}")

        raise RuntimeError(
            f"{len(failed)} channel set(s) failed to build."
        )
=======
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
>>>>>>> 35a5e30607919fdd4a607b3fe50e782148c56d3d


if __name__ == "__main__":
    main()