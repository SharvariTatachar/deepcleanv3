import json
import os
import re
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path


RUN_TYPES = ("hyperparam", "architecture", "other")
METADATA_NAME = "metadata.json"

HYPERPARAM_KEYS = (
    "lr",
    "cnn_lr",
    "transformer_lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "num_workers",
    "fftlength",
    "overlap",
    "psd_weight",
    "mse_weight",
    "cross_psd_weight",
    "edge_weight",
    "edge_frac",
    "train_kernel",
    "train_stride",
    "filt_fl",
    "filt_fh",
    "filt_order",
    "filt_pad",
    "pad_mode",
    "fs",
    "seed",
    "train_frac",
)


def add_run_args(parser):
    parser.add_argument(
        "--run-type",
        type=str,
        choices=list(RUN_TYPES),
        default="other",
        help="Group this run under runs/{hyperparam,architecture,other}",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional slug used when auto-naming the run directory",
    )


def _slugify(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return text[:80]


def default_slug(params):
    if getattr(params, "run_name", None):
        return _slugify(params.run_name)

    parts = []
    chanslist = getattr(params, "chanslist", None)
    if chanslist:
        parts.append(_slugify(Path(chanslist).stem))
    seed = getattr(params, "seed", None)
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(parts) if parts else None


def resolve_train_dir(params, model_key):
    if getattr(params, "train_dir", None):
        Path(params.train_dir).mkdir(parents=True, exist_ok=True)
        return params.train_dir

    run_type = getattr(params, "run_type", None) or "other"
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}, got {run_type!r}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{stamp}_{_slugify(model_key)}"
    slug = default_slug(params)
    if slug:
        name = f"{name}_{slug}"

    train_dir = Path("runs") / run_type / name
    train_dir.mkdir(parents=True, exist_ok=True)
    return str(train_dir)


def jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def params_dict(params):
    skip = {"log"}
    out = {}
    for key, value in vars(params).items():
        if key in skip:
            continue
        out[key] = jsonable(value)
    return out


def hyperparams_from_params(params):
    out = {}
    for key in HYPERPARAM_KEYS:
        if hasattr(params, key):
            out[key] = jsonable(getattr(params, key))
    return out


def model_config(model):
    cfg = {
        "class_name": model.__class__.__name__,
        "module": type(model).__module__,
    }
    for key, value in vars(model).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            cfg[key] = value
    n_params = sum(int(p.numel()) for p in model.parameters())
    cfg["n_parameters"] = n_params
    return cfg


def metadata_path(train_dir):
    return os.path.join(train_dir, METADATA_NAME)


def load_metadata(train_dir):
    path = metadata_path(train_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_metadata(train_dir, data):
    os.makedirs(train_dir, exist_ok=True)
    path = metadata_path(train_dir)
    with open(path, "w") as f:
        json.dump(jsonable(data), f, indent=2)
    return path


def update_metadata(train_dir, **updates):
    data = load_metadata(train_dir)
    data.update(updates)
    return write_metadata(train_dir, data)


def copy_config(config_path, train_dir):
    if not config_path:
        return None
    src = Path(config_path)
    if not src.is_file():
        return None
    dest = Path(train_dir) / "config.ini"
    shutil.copy2(src, dest)
    return str(dest)


def build_metadata(params, model_name, extra=None):
    now = datetime.now().astimezone()
    data = {
        "run_name": os.path.basename(params.train_dir),
        "run_type": getattr(params, "run_type", "other"),
        "status": "running",
        "datetime": now.isoformat(timespec="seconds"),
        "unix_time": int(now.timestamp()),
        "hostname": socket.gethostname(),
        "model_name": model_name,
        "train_dir": params.train_dir,
        "config": getattr(params, "config", None),
        "cli": list(sys.argv),
        "model": {"name": model_name},
        "hyperparameters": hyperparams_from_params(params),
        "data": {
            "dataset_file": getattr(params, "dataset_file", None),
            "chanslist": getattr(params, "chanslist", None),
            "channel_set_json": getattr(params, "channel_set_json", None),
            "train_t0": getattr(params, "train_t0", None),
            "train_duration": getattr(params, "train_duration", None),
            "seed": getattr(params, "seed", None),
        },
        "params": params_dict(params),
    }
    if extra:
        data.update(extra)
    return data


def start_run(params, model_name, extra=None):
    copy_config(getattr(params, "config", None), params.train_dir)
    return write_metadata(
        params.train_dir,
        build_metadata(params, model_name, extra=extra),
    )


def finish_run(params, history=None, model=None, extra=None):
    updates = {
        "status": "completed",
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if history is not None:
        updates["history"] = history
    if model is not None:
        updates["model"] = model_config(model)
        updates["model_name"] = model.__class__.__name__
    if extra:
        updates.update(extra)
    return update_metadata(params.train_dir, **updates)
