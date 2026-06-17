import os
import sys
import logging

logger = logging.getLogger(__name__)

import numpy as np

import torch

from ..logger import Logger

# TODO: add more choices for noisy channels 

def select_chans(all_chans, baseline_chans, noisy_pool, noisy_choices=[5]): 
    "Keep baseline channels for each epoch, randomly select noisy channels"
    n_noisy = np.random.choice(noisy_choices)
    
    # sample noisy channels using n_noisy
    sampled_noisy = list(
        np.random.choice(
            noisy_pool,
            size=min(n_noisy, len(noisy_pool)),
            replace=False
        )
    )

    selected_names = baseline_chans + sampled_noisy

    # mapping channel names to indices 
    channel_to_idx = {
        ch: i for i, ch in enumerate(all_chans)
    }

    selected_idx = [
        channel_to_idx[ch]
        for ch in selected_names
    ]

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
    # If Logger is not given, create a default logger
    if logger is None:
        logger = Logger(outdir="train_dir", metrics=["loss"])

    num_batches = len(train_loader)
    model.train()

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(max_epochs):
        model.train()

        if dynamic_channels:
            selected_idx, selected_names, sampled_noisy = select_chans(
                all_channels,
                baseline_channels,
                noisy_pool,
            )

            selected_idx_tensor = torch.as_tensor(
                selected_idx,
                dtype=torch.long,
                device=device,
            )

            selected_ids = torch.tensor(
                [channel_to_id[ch] for ch in selected_names],
                dtype=torch.long,
                device=device,
            )

        else:
            selected_idx_tensor = None

            if fixed_channel_ids is None:
                raise ValueError(
                    "fixed_channel_ids must be provided when dynamic_channels=False"
                )

            selected_ids = fixed_channel_ids.to(device)

        train_loss_sum = torch.zeros((), device=device)
        n_seen = 0

        for step, (x, tgt) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            if dynamic_channels:
                if step == 0:
                    assert selected_idx_tensor.max().item() < x.shape[1], (
                        f"Bad channel index: max idx {selected_idx_tensor.max().item()} "
                        f"but x has only {x.shape[1]} channels"
                    )

                x = x[:, selected_idx_tensor, :]

            channel_ids = selected_ids.unsqueeze(0).expand(x.size(0), -1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x, channel_ids=channel_ids).squeeze(1)
                loss = criterion(pred, tgt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if hasattr(criterion, "reduction") and criterion.reduction == "mean":
                train_loss_sum += loss.detach() * x.size(0)
                n_seen += x.size(0)
            else:
                train_loss_sum += loss.detach()
                n_seen += 1

        train_loss = (train_loss_sum / n_seen).item()

        val_loss = 0.0

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

                    channel_ids_val = selected_ids.unsqueeze(0).expand(
                        x_val.size(0), -1
                    )

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        pred_val = model(
                            x_val,
                            channel_ids=channel_ids_val,
                        ).squeeze(1)

                        loss_val = criterion(pred_val, tgt_val)

                    if hasattr(criterion, "reduction") and criterion.reduction == "mean":
                        val_loss_sum += loss_val.detach() * x_val.size(0)
                        n_val += x_val.size(0)
                    else:
                        val_loss_sum += loss_val.detach()
                        n_val += 1

            val_loss = (val_loss_sum / n_val).item()
            model.train()

        if lr_scheduler is not None:
            lr_scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        last_batch_idx = num_batches - 1

        logger.update_metric(
            train_loss,
            val_loss,
            "loss",
            epoch,
            last_batch_idx,
            num_batches,
        )

        logger.display_status(
            epoch,
            max_epochs,
            last_batch_idx,
            num_batches,
            train_loss,
            val_loss,
            "loss",
        )

        logger.log_metric()
        logger.save_model(model, epoch)

    logging.info(f"Training completed. Final train loss: {train_loss:.6f}")

    return history

def get_device(device):
    ''' Convenient function to set up hardware '''
    if device.lower() == 'cpu':
        device = torch.device('cpu')
    elif device.lower() == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info(f'-Use device: {device}')
        else:
            logging.warning('No MPS available. Use CPU instead.')
            device = torch.device('cpu')
    elif 'cuda' in device.lower():
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            logging.warning('No GPU available. Use CPU instead.')
            device = torch.device('cpu')
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory *= 1e-9 # convert bytes to Gb
        logger.info('- Use device: {}'.format(torch.cuda.get_device_name(device)))
        logger.info("device count: {}".format(torch.cuda.device_count()))
        logger.info('- Total memory: {:.4f} GB'.format(total_memory))
    else:
        logger.info('- Use device: CPU')
    return device

