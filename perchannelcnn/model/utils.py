import logging

import numpy as np
import torch

from ..logger import Logger

logger = logging.getLogger(__name__)


def select_chans(all_chans, baseline_chans, noisy_pool, noisy_choices=(5,)):
    """Keep baseline channels and randomly select additional witnesses."""
    n_noisy = int(np.random.choice(noisy_choices))
    sampled_noisy = list(np.random.choice(noisy_pool, size=min(n_noisy, len(noisy_pool)), replace=False))
    selected_names = list(baseline_chans) + sampled_noisy
    channel_to_idx = {ch: i for i, ch in enumerate(all_chans)}
    selected_idx = [channel_to_idx[ch] for ch in selected_names]
    return selected_idx, selected_names, sampled_noisy


def train(
    train_loader,
    model,
    criterion,
    device,
    optimizer,
    lr_scheduler,
    val_loader=None,
    max_epochs=10,
    logger=None,
    dynamic_channels=False,
    all_channels=None,
    baseline_channels=None,
    noisy_pool=None,
    channel_to_id=None,
    fixed_channel_ids=None,
):
    """Train PCC using learned channel IDs and the original DeepClean loop."""
    if logger is None:
        logger = Logger(outdir="train_dir", metrics=["loss"])

    num_batches = len(train_loader)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()

        if dynamic_channels:
            if any(v is None for v in (all_channels, baseline_channels, noisy_pool, channel_to_id)):
                raise ValueError("Dynamic channels require all_channels, baseline_channels, noisy_pool, and channel_to_id")
            selected_idx, selected_names, _ = select_chans(all_channels, baseline_channels, noisy_pool)
            selected_idx_tensor = torch.as_tensor(selected_idx, dtype=torch.long, device=device)
            selected_ids = torch.as_tensor([channel_to_id[ch] for ch in selected_names], dtype=torch.long, device=device)
        else:
            selected_idx_tensor = None
            if fixed_channel_ids is not None:
                selected_ids = torch.as_tensor(fixed_channel_ids, dtype=torch.long, device=device)
            elif baseline_channels is not None and channel_to_id is not None:
                selected_ids = torch.as_tensor([channel_to_id[ch] for ch in baseline_channels], dtype=torch.long, device=device)
            else:
                selected_ids = None  # PCC forward defaults to arange(C)

        train_loss_sum = torch.zeros((), device=device)
        n_seen = 0

        for step, (x, tgt) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            if dynamic_channels:
                if step == 0 and (selected_idx_tensor.min().item() < 0 or selected_idx_tensor.max().item() >= x.shape[1]):
                    raise ValueError("Selected channel index is outside input channel dimension")
                x = x[:, selected_idx_tensor, :]

            if selected_ids is not None and selected_ids.numel() != x.shape[1]:
                raise ValueError(f"{selected_ids.numel()} channel IDs for {x.shape[1]} input channels")
            channel_ids = None if selected_ids is None else selected_ids.unsqueeze(0).expand(x.size(0), -1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(x, channel_ids=channel_ids).squeeze(1)
                loss = criterion(pred, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if getattr(criterion, "reduction", None) == "mean":
                train_loss_sum += loss.detach() * x.size(0)
                n_seen += x.size(0)
            else:
                train_loss_sum += loss.detach()
                n_seen += 1

        if n_seen == 0:
            raise RuntimeError("No training batches were processed")
        train_loss = (train_loss_sum / n_seen).item()

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_loss_sum = torch.zeros((), device=device)
            n_val = 0
            with torch.no_grad():
                for x_val, tgt_val in val_loader:
                    x_val = x_val.to(device, non_blocking=True)
                    tgt_val = tgt_val.to(device, non_blocking=True)
                    if dynamic_channels:
                        x_val = x_val[:, selected_idx_tensor, :]
                    channel_ids_val = None if selected_ids is None else selected_ids.unsqueeze(0).expand(x_val.size(0), -1)
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        pred_val = model(x_val, channel_ids=channel_ids_val).squeeze(1)
                        loss_val = criterion(pred_val, tgt_val)
                    if getattr(criterion, "reduction", None) == "mean":
                        val_loss_sum += loss_val.detach() * x_val.size(0)
                        n_val += x_val.size(0)
                    else:
                        val_loss_sum += loss_val.detach()
                        n_val += 1
            if n_val == 0:
                raise RuntimeError("No validation batches were processed")
            val_loss = (val_loss_sum / n_val).item()
            model.train()

        if lr_scheduler is not None:
            lr_scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        last_batch_idx = num_batches - 1
        logger.update_metric(train_loss, val_loss, "loss", epoch, last_batch_idx, num_batches)
        logger.display_status(epoch, max_epochs, last_batch_idx, num_batches, train_loss, val_loss, "loss")
        logger.log_metric()
        logger.save_model(model, epoch)

    logging.info("Training completed. Final train loss: %.6f", train_loss)
    return history


def get_device(device):
    """Convenience function for selecting the compute device."""
    device = device or "cuda"
    if device.lower() == "cpu":
        result = torch.device("cpu")
    elif device.lower() == "mps":
        if torch.backends.mps.is_available():
            result = torch.device("mps")
        else:
            logging.warning("No MPS available. Use CPU instead.")
            result = torch.device("cpu")
    elif "cuda" in device.lower():
        if torch.cuda.is_available():
            result = torch.device(device)
        else:
            logging.warning("No GPU available. Use CPU instead.")
            result = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device: {device}")

    if result.type == "cuda":
        total_memory = torch.cuda.get_device_properties(result).total_memory * 1e-9
        logger.info("- Use device: %s", torch.cuda.get_device_name(result))
        logger.info("device count: %s", torch.cuda.device_count())
        logger.info("- Total memory: %.4f GB", total_memory)
    else:
        logger.info("- Use device: %s", result)
    return result
