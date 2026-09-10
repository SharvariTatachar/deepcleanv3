import json
import subprocess
from pathlib import Path

import optuna


def load_objective_value(run_dir):
    json_path = run_dir / "dc_pcc_run.json"
    if not json_path.exists():
        json_path = run_dir / "dc_transform_run.json"

    with open(json_path) as f:
        data = json.load(f)

    history = data["history"]

    val = (
        history.get("test_loss")
        or history.get("val_loss")
        or history.get("valid_loss")
    )

    if val is None:
        print("Available history keys:", history.keys())
        raise KeyError("Could not find validation/test loss")

    return min(val)


def objective(trial):
    # Small, focused search space
    wd = trial.suggest_categorical(
        "weight_decay",
        [1e-4, 3e-4, 1e-3],
    )
    
    # Keep LR fixed for now
    cnn_lr = 1e-3
    transformer_lr = 1e-3

    run_dir = Path(f"runs_wd_sweep_stridefix/trial_{trial.number}_wd{wd:.0e}")
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "dc_pcc_run.json"

    if json_path.exists():
        print(f"Trial {trial.number} already exists. Reusing {json_path}.")
    else:
        cmd = [
            "python3", "dc-transform-train.py",
            "--config", "configs/118dcv3train.ini",
            "--chanslist", "channelconfigs/add150low.ini",
            "--dataset-file", "data/channel_ablation_npzs/add150low.npz",
            "--train-t0", "0",
            "--seed", "0",
            "--max-epochs", "50",
            "--cnn-lr", str(cnn_lr),
            "--transformer-lr", str(transformer_lr),
            "--weight-decay", str(wd),
            "--train-dir", str(run_dir),
            "--log", "train.log",
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    return load_objective_value(run_dir)

search_space = {
    "weight_decay": [ 1e-4, 3e-4, 1e-3] ,
}
sampler = optuna.samplers.GridSampler(search_space)

study = optuna.create_study(
    sampler=sampler,
    direction="minimize",
    study_name="pcc_stridefix_wd_1e4_to_1e3",
    storage="sqlite:///pcc_stridefix_wd_1e4_to_1e3.db",
    load_if_exists=True,
)

study.optimize(objective, n_trials=3)

print("Best value:", study.best_value)
print("Best params:", study.best_params)

df = study.trials_dataframe()
df.to_csv("pcc_stridefix_wd_1e4_to_1e3_results.csv", index=False)
print("Saved results to pcc_stridefix_wd_1e4_to_1e3_results.csv")