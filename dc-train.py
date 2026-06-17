import os
import torch
import time
import pickle
import argparse
import json
import configparser
import logging
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

import deepclean as dc
import deepclean.timeseries as ts
import deepclean.criterion
import deepclean.model.utils as utils
import deepclean.model.deepclean


TARGET_CHANNEL = "H1:GDS-CALIB_STRAIN"


def normalize_channel_name(ch):
    return (
        ch.replace(":", "_")
          .replace("-", "_")
          .replace(".", "_")
          .upper()
    )


def get_channel_name_from_file(npz_path):
    """
    Converts:
      H1_PEM_EX_SEIS_VEA_FLOOR_Y_DQ.npz
    to:
      H1:PEM_EX_SEIS_VEA_FLOOR_Y_DQ
    """
    stem = Path(npz_path).stem
    if "_" not in stem:
        raise ValueError(f"Unexpected filename format: {npz_path}")

    ifo, rest = stem.split("_", 1)
    return f"{ifo}:{rest}"


def load_single_channel_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        if "ts" not in data:
            raise ValueError(
                f"{npz_path} does not contain key 'ts'. "
                f"Found keys: {list(data.keys())}"
            )
        return data["ts"]


def make_combined_npz_from_random_files(
    dataset_dir,
    out_npz,
    out_channels_ini,
    target_channel=TARGET_CHANNEL,
    n_witnesses=35,
    seed=0,
):
    dataset_dir = Path(dataset_dir)
    rng = random.Random(seed)

    npz_files = sorted(dataset_dir.glob("*.npz"))

    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {dataset_dir}")

    channel_to_file = {}
    for f in npz_files:
        ch = get_channel_name_from_file(f)
        channel_to_file[ch] = f

    # Robust target matching
    if target_channel not in channel_to_file:
        target_norm = normalize_channel_name(target_channel)

        matches = [
            ch for ch in channel_to_file
            if normalize_channel_name(ch) == target_norm
        ]

        if len(matches) == 1:
            old_target = target_channel
            target_channel = matches[0]
            logging.info(f"Matched target alias {old_target} -> {target_channel}")
        else:
            print("\nCould not find exact target.")
            print("Looking for:", repr(target_channel))
            print("\nPossible GDS/CALIB/STRAIN files:")
            for ch in channel_to_file:
                if "GDS" in ch or "CALIB" in ch or "STRAIN" in ch:
                    print(" ", repr(ch), "->", channel_to_file[ch])

            raise ValueError(
                f"Target channel {target_channel} not found in {dataset_dir}"
            )

    witness_pool = [
        ch for ch in channel_to_file
        if ch != target_channel
    ]

    if n_witnesses > len(witness_pool):
        raise ValueError(
            f"Requested {n_witnesses} witnesses, "
            f"but only found {len(witness_pool)}"
        )

    selected_witnesses = rng.sample(witness_pool, n_witnesses)
    selected_channels = [target_channel] + selected_witnesses

    combined = {}
    for ch in selected_channels:
        f = channel_to_file[ch]
        combined[ch] = load_single_channel_npz(f)

    out_npz = str(out_npz)
    out_channels_ini = str(out_channels_ini)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, **combined)

    with open(out_channels_ini, "w") as f:
        for ch in selected_channels:
            f.write(ch + "\n")

    metadata = {
        "dataset_dir": str(dataset_dir),
        "combined_npz": out_npz,
        "channels_ini": out_channels_ini,
        "target": target_channel,
        "n_witnesses": n_witnesses,
        "seed": seed,
        "selected_witnesses": selected_witnesses,
        "selected_channels": selected_channels,
        "n_total_npz_files": len(npz_files),
        "n_total_witness_files": len(witness_pool),
    }

    metadata_path = out_npz.replace(".npz", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nCreated random combined dataset:")
    print("  combined npz:", out_npz)
    print("  channels ini:", out_channels_ini)
    print("  metadata:", metadata_path)
    print("  target:", target_channel)
    print("  n witnesses:", len(selected_witnesses))

    return out_npz, out_channels_ini, metadata


def load_channels(path):
    with open(path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def make_channels_ini_from_json(channel_set_json, out_path):
    with open(channel_set_json, "r") as f:
        cfg = json.load(f)

    target = cfg["target"]
    channels = cfg["channels"]

    lines = [target] + [ch for ch in channels if ch != target]

    with open(out_path, "w") as f:
        for ch in lines:
            f.write(ch + "\n")

    return out_path, cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("yes", "true", "t", "1")
    return bool(v)


def parse_cmd():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        usage="%(prog)s [options]"
    )

    parser.add_argument("--config", type=str, default="configs/110train.ini")

    parser.add_argument("--train-t0", type=int, default=None)
    parser.add_argument("--train-duration", type=int, default=None)
    parser.add_argument("--chanslist", type=str, default=None)
    parser.add_argument("--fs", type=float, default=None)
    parser.add_argument("--channel-set-json", type=str, default=None)

    parser.add_argument(
        "--dataset-file",
        type=str,
        default="/storage/home/hcoda1/3/statachar3/r-pli77-0/deepcleanv3/data/combined_data_updated.npz",
        help="Path to combined training dataset npz"
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/storage/home/hcoda1/3/statachar3/r-pli77-0/deepcleanv3/data/dataset",
        help="Directory containing separate single-channel npz files"
    )

    parser.add_argument("--random-channels", type=str2bool, default=False)
    parser.add_argument("--n-random-witnesses", type=int, default=35)

    parser.add_argument(
        "--random-subset-npz",
        type=str,
        default=None,
        help="Output path for generated combined random-channel npz"
    )

    parser.add_argument("--filt-fl", nargs="+", type=float, default=None)
    parser.add_argument("--filt-fh", nargs="+", type=float, default=None)
    parser.add_argument("--filt-order", type=int, default=None)
    parser.add_argument("--filt-pad", type=float, default=5.0)

    parser.add_argument("--train-kernel", type=float, default=None)
    parser.add_argument("--train-stride", type=float, default=None)
    parser.add_argument("--pad-mode", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    parser.add_argument("--fftlength", type=float, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--psd-weight", type=float, default=None)
    parser.add_argument("--mse-weight", type=float, default=None)

    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--load-dataset", default=None, type=str2bool)
    parser.add_argument("--save-dataset", default=None, type=str2bool)
    parser.add_argument("--initial-checkpoint", default=None, type=str)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    params = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(params.config)
    c = cfg["config"]

    if params.train_dir is None and "train_dir" in c:
        params.train_dir = c.get("train_dir")
    if params.log is None and "log" in c:
        params.log = c.get("log")
    if params.device is None and "device" in c:
        params.device = c.get("device")
    if params.train_t0 is None and "train_t0" in c:
        params.train_t0 = c.getint("train_t0")
    if params.train_duration is None and "train_duration" in c:
        params.train_duration = c.getint("train_duration")
    if params.fs is None and "fs" in c:
        params.fs = c.getint("fs")
    if params.chanslist is None and "chanslist" in c:
        params.chanslist = c.get("chanslist")
    if params.train_kernel is None and "train_kernel" in c:
        params.train_kernel = c.getfloat("train_kernel")
    if params.train_stride is None and "train_stride" in c:
        params.train_stride = c.getfloat("train_stride")
    if params.pad_mode is None and "pad_mode" in c:
        params.pad_mode = c.get("pad_mode")
    if params.filt_fl is None and "filt_fl" in c:
        params.filt_fl = [c.getfloat("filt_fl")]
    if params.filt_fh is None and "filt_fh" in c:
        params.filt_fh = [c.getfloat("filt_fh")]
    if params.filt_order is None and "filt_order" in c:
        params.filt_order = c.getint("filt_order")
    if params.batch_size is None and "batch_size" in c:
        params.batch_size = c.getint("batch_size")
    if params.max_epochs is None and "max_epochs" in c:
        params.max_epochs = c.getint("max_epochs")
    if params.num_workers is None and "num_workers" in c:
        params.num_workers = c.getint("num_workers")
    if params.lr is None and "lr" in c:
        params.lr = c.getfloat("lr")
    if params.weight_decay is None and "weight_decay" in c:
        params.weight_decay = c.getfloat("weight_decay")
    if params.fftlength is None and "fftlength" in c:
        params.fftlength = c.getfloat("fftlength")
    if params.overlap is None and "overlap" in c:
        params.overlap = c.getfloat("overlap")
    if params.psd_weight is None and "psd_weight" in c:
        params.psd_weight = c.getfloat("psd_weight")
    if params.mse_weight is None and "mse_weight" in c:
        params.mse_weight = c.getfloat("mse_weight")
    if params.save_dataset is None and "save_dataset" in c:
        params.save_dataset = c.getboolean("save_dataset")
    if params.load_dataset is None and "load_dataset" in c:
        params.load_dataset = c.getboolean("load_dataset")

    if params.train_dir is None:
        params.train_dir = "."
    if params.fs is None:
        params.fs = 2048
    if params.overlap is None:
        params.overlap = 0.5

    return params


timestamp = int(time.time())
params = parse_cmd()
set_seed(params.seed)

print(f"Using seed: {params.seed}")

# pickle.dump({"params": params}, open("dc_transform_train.p", "wb"))
# params = pickle.load(open("dc_transform_train.p", "rb"))["params"]

os.makedirs(params.train_dir, exist_ok=True)

if params.log is not None:
    params.log = os.path.join(params.train_dir, params.log)

if params.log is not None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(params.log, mode="a"),
            logging.StreamHandler()
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

logging.info(f"Create training directory: {params.train_dir}")

device = utils.get_device(params.device)

train_data = ts.TimeSeriesSegmentDataset(
    params.train_kernel,
    params.train_stride,
    pad_mode=params.pad_mode or "median",
    fs=params.fs,
)

val_data = ts.TimeSeriesSegmentDataset(
    kernel=params.train_kernel,
    stride=params.train_stride,
    pad_mode=params.pad_mode or "median",
    fs=params.fs,
)

if params.random_channels:
    channels_file = os.path.join(params.train_dir, "channels_random.ini")

    if params.random_subset_npz is None:
        params.random_subset_npz = os.path.join(
            params.train_dir,
            f"combined_random_{params.n_random_witnesses}_seed{params.seed}.npz"
        )

    params.dataset_file, channels_file, channel_cfg = make_combined_npz_from_random_files(
        dataset_dir=params.dataset_dir,
        out_npz=params.random_subset_npz,
        out_channels_ini=channels_file,
        target_channel=TARGET_CHANNEL,
        n_witnesses=params.n_random_witnesses,
        seed=params.seed,
    )

    logging.info(f"Created combined random npz: {params.dataset_file}")
    logging.info(f"Using generated channel file: {channels_file}")

elif params.channel_set_json is not None:
    run_channel_file = os.path.join(params.train_dir, "channels_selected.ini")
    channels_file, channel_cfg = make_channels_ini_from_json(
        params.channel_set_json,
        run_channel_file
    )

else:
    channels_file = params.chanslist or "ogchannels.ini"
    channel_cfg = None


# Read train/validation data
train_data.read(
    params.dataset_file,
    channels=channels_file,
    start_time=params.train_t0,
    end_time=params.train_t0 + 1536,
    fs=params.fs,
)

val_data.read(
    params.dataset_file,
    channels=channels_file,
    start_time=params.train_t0 + 1536,
    end_time=params.train_t0 + 3072,
    fs=params.fs,
)

all_channels = [
    ch for ch in load_channels(channels_file)
    if ch != TARGET_CHANNEL
]

channel_to_id = {
    ch: i for i, ch in enumerate(all_channels)
}

# Bandpass target only
train_data = train_data.bandpass(
    params.filt_fl,
    params.filt_fh,
    params.filt_order,
    channels="target",
)

val_data = val_data.bandpass(
    params.filt_fl,
    params.filt_fh,
    params.filt_order,
    channels="target",
)

# Trim filter padding
pad = int(params.filt_pad * params.fs)
if pad > 0:
    train_data.data = train_data.data[:, pad:-pad]
    val_data.data = val_data.data[:, pad:-pad]

# Normalize using train stats
mean = train_data.mean
std = train_data.std

train_data = train_data.normalize()
val_data = val_data.normalize(mean, std)

# Rebuild windows after trimming/filtering
train_data.build_windows()
val_data.build_windows()

logging.info(f"Train windows: {len(train_data)}")
logging.info(f"Val windows: {len(val_data)}")
logging.info(f"Number of channels including target: {train_data.n_channels}")
logging.info(f"Number of witness channels: {train_data.n_channels - 1}")

g = torch.Generator()
g.manual_seed(params.seed)

num_workers = params.num_workers if params.num_workers is not None else 4

loader_kwargs = dict(
    batch_size=params.batch_size,
    shuffle=False,
    generator=g,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    pin_memory=True,
    drop_last=True,
)

if num_workers > 0:
    loader_kwargs.update(
        persistent_workers=True,
        prefetch_factor=4,
    )

train_loader = DataLoader(train_data, **loader_kwargs)
val_loader = DataLoader(val_data, **loader_kwargs)

x, tgt = next(iter(train_loader))
logging.info(f"Batch x shape: {x.shape}")
logging.info(f"Batch target shape: {tgt.shape}")

# Original DeepClean model
model = dc.model.deepclean.DeepClean(train_data.n_channels - 1)
model = model.to(device)

criterion = dc.criterion.CompositePSDLoss(
    fs=params.fs,
    fl=params.filt_fl,
    fh=params.filt_fh,
    fftlength=params.fftlength,
    overlap=params.overlap,
    psd_weight=params.psd_weight,
    mse_weight=params.mse_weight,
    reduction="mean",
    device=device,
    average="mean",
)

optimizer = optim.Adam(
    model.parameters(),
    lr=params.lr,
    weight_decay=params.weight_decay,
)

lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1,
)

train_logger = dc.logger.Logger(
    outdir=params.train_dir,
    metrics=["loss"],
)

history = utils.train(
    train_loader,
    model,
    criterion,
    device,
    optimizer,
    lr_scheduler,
    val_loader=val_loader,
    max_epochs=params.max_epochs,
    logger=train_logger,
    dynamic_channels=False,
    all_channels=all_channels,
    baseline_channels=all_channels,
    noisy_pool=None,
    channel_to_id=channel_to_id,
)

run_data = {
    "model_name": model.__class__.__name__,
    "batch_size": params.batch_size,
    "lr": params.lr,
    "weight_decay": params.weight_decay,
    "max_epochs": params.max_epochs,
    "train_t0": params.train_t0,
    "train_duration": params.train_duration,
    "fs": params.fs,
    "filt_fl": params.filt_fl,
    "filt_fh": params.filt_fh,
    "dataset_file": params.dataset_file,
    "channels_file": channels_file,
    "random_channels": params.random_channels,
    "n_random_witnesses": params.n_random_witnesses,
    "channel_set_json": params.channel_set_json,
    "channel_set_metadata": channel_cfg,
    "history": history,
}

run_path = os.path.join(params.train_dir, "dc_run.json")
with open(run_path, "w") as f:
    json.dump(run_data, f, indent=2)

logging.info(f"Saved run data to {run_path}")