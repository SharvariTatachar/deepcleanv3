import os
import sys
import logging

logger = logging.getLogger(__name__)

import numpy as np
import torch

from ..logger import Logger

def get_grad_norm(parameters, norm_type=2.0):
    """Compute total gradient norm without modifying gradients."""
    grads = [
        p.grad.detach()
        for p in parameters
        if p.grad is not None
    ]

    if not grads:
        return 0.0

    total_norm = torch.norm(
        torch.stack([
            torch.norm(g, norm_type)
            for g in grads
        ]),
        norm_type,
    )

    return total_norm.item()

def select_chans(all_chans, baseline_chans, noisy_pool, noisy_choices=[20]):
    """Keep baseline channels and randomly select noisy channels."""
    n_noisy = np.random.choice(noisy_choices)

    sampled_noisy = list(
        np.random.choice(
            noisy_pool,
            size=min(n_noisy, len(noisy_pool)),
            replace=False,
        )
    )

    selected_names = baseline_chans + sampled_noisy

    channel_to_idx = {ch: i for i, ch in enumerate(all_chans)}

    selected_idx = [channel_to_idx[ch] for ch in selected_names]

    return selected_idx, selected_names, sampled_noisy


def attention_entropy(attn, eps=1e-8, normalize=True):
    """
    attn: [B*Tds, nhead, C, C]
    Returns scalar entropy.

    Normalized entropy:
      ~1.0 = uniform attention
      ~0.0 = sharp/selective attention
    """
    p = attn.clamp_min(eps)

    ent = -(p * p.log()).sum(dim=-1)

    if normalize:
        C = attn.shape[-1]
        ent = ent / torch.log(
            torch.tensor(C, device=attn.device, dtype=attn.dtype)
        )

    return ent.mean()


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
    grad_clip_norm=2.0,
):
    if logger is None:
        logger = Logger(outdir="train_dir", metrics=["loss"])

    num_batches = len(train_loader)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "attn_entropy": [],
        "transformer_alpha": [],
        "learning_rate": [],
        "transformer_learning_rate": [],
        "gradient_norm": [],
        "base_gradient_norm": [], 
        "transformer_gradient_norm": [], 
        "correction_gradient_norm": [], 
        "pcc_norm": [],
        "correction_norm": [],
        "scaled_correction_norm": [],
        "raw_correction_ratio": [],
        "correction_ratio": [],
        "correction_gate": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1

    patience = 100
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # ---------------------------------------------------------
        # Tell HybridTransformerCNN which epoch is currently running
        # ---------------------------------------------------------
        model_core = model.module if hasattr(model, "module") else model

        model_core.set_epoch(epoch)
        transformer_alpha = model_core.get_transformer_alpha()

        correction_head_trainable = (
            epoch >= model_core.correction_head_freeze_epochs
        )
        model_core.set_correction_head_trainable(
            correction_head_trainable
        )

        model.train()

        # Optimizer group 0 = base/CNN parameters.
        # If a second group exists, it is the transformer group.
        current_lr = optimizer.param_groups[0]["lr"]
        transformer_lr = (
            optimizer.param_groups[1]["lr"]
            if len(optimizer.param_groups) > 1
            else current_lr
        )

        logging.info(
            "Starting epoch %d/%d: alpha=%.4f, "
            "correction_head=%s, base_lr=%.3e, transformer_lr=%.3e",
            epoch + 1,
            max_epochs,
            transformer_alpha,
            "trainable" if correction_head_trainable else "frozen",
            current_lr,
            transformer_lr,
        )

        entropy_sum = 0.0
        entropy_count = 0

        grad_norm_values = []
        base_grad_norm_values = [] 
        transformer_grad_norm_values = []
        correction_grad_norm_values = [] 
    
        clipped_count = 0


        # ---------------------------------------------------------
        # Transformer/correction diagnostics.
        # Accumulate TRAINING batches only so validation forwards do
        # not overwrite the epoch statistics.
        # ---------------------------------------------------------
        alpha_sum = 0.0
        gate_sum = 0.0
        pcc_norm_sum = 0.0
        correction_norm_sum = 0.0
        scaled_correction_norm_sum = 0.0
        raw_correction_ratio_sum = 0.0
        correction_ratio_sum = 0.0
        diagnostic_count = 0

        # ---------------------------------------------------------
        # Select channels once for this epoch
        # ---------------------------------------------------------
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

            selected_ids = torch.as_tensor(
                [channel_to_id[ch] for ch in selected_names],
                dtype=torch.long,
                device=device,
            )

            logging.info(
                "Epoch %d selected %d channels: %d baseline + %d noisy",
                epoch + 1,
                len(selected_names),
                len(baseline_channels),
                len(sampled_noisy),
            )

        else:
            selected_idx_tensor = None

            if fixed_channel_ids is None:
                raise ValueError(
                    "fixed_channel_ids must be provided when "
                    "dynamic_channels=False"
                )

            selected_ids = fixed_channel_ids.to(
                device=device,
                dtype=torch.long,
            )

        train_loss_sum = torch.zeros((), device=device)
        n_seen = 0

        # ---------------------------------------------------------
        # Training
        # ---------------------------------------------------------
        for step, (x, tgt) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            if dynamic_channels:
                if step == 0:
                    assert selected_idx_tensor.max().item() < x.shape[1], (
                        f"Bad channel index: maximum index "
                        f"{selected_idx_tensor.max().item()}, but x only has "
                        f"{x.shape[1]} channels."
                    )

                x = x[:, selected_idx_tensor, :]

            channel_ids = selected_ids.unsqueeze(0).expand(
                x.size(0),
                -1,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                pred = model(
                    x,
                    channel_ids=channel_ids,
                ).squeeze(1)

            # Read diagnostics from this TRAINING forward pass.
            diag_values = (
                model_core.last_transformer_alpha,
                model_core.last_correction_gate,
                model_core.last_pcc_norm,
                model_core.last_raw_correction_norm,
                model_core.last_scaled_correction_norm,
                model_core.last_raw_correction_ratio,
                model_core.last_applied_correction_ratio,
            )

            if all(
                value is not None and np.isfinite(float(value))
                for value in diag_values
            ):
                (
                    batch_alpha,
                    batch_gate,
                    batch_pcc_norm,
                    batch_corr_norm,
                    batch_scaled_corr_norm,
                    batch_raw_corr_ratio,
                    batch_applied_corr_ratio,
                ) = map(float, diag_values)

                alpha_sum += batch_alpha
                gate_sum += batch_gate
                pcc_norm_sum += batch_pcc_norm
                correction_norm_sum += batch_corr_norm
                scaled_correction_norm_sum += batch_scaled_corr_norm

                raw_correction_ratio_sum += batch_raw_corr_ratio
                correction_ratio_sum += batch_applied_corr_ratio

                diagnostic_count += 1

                if epoch == 0 and step == 0:
                    print(
                        "INSIDE TRAIN pred shape:",
                        pred.shape,
                        flush=True,
                    )
                    print(
                        "INSIDE TRAIN tgt shape:",
                        tgt.shape,
                        flush=True,
                    )

                loss = criterion(pred, tgt)

            # -----------------------------------------------------
            # Backpropagation with AMP-safe gradient clipping
            # -----------------------------------------------------
            scaler.scale(loss).backward()

            # Gradients must be unscaled before clipping.
            scaler.unscale_(optimizer)
            # PCC / base network
            base_params = list(model.downsample.parameters()) + \
                        list(model.upsample.parameters()) + \
                        list(model.channel_embedding.parameters())

            base_grad_norm = get_grad_norm(base_params)

            # Transformer itself
            transformer_grad_norm = get_grad_norm(
                model.transformer.parameters()
            )

            # Correction head
            correction_grad_norm = get_grad_norm(
                model.correction_head.parameters()
            )

            if np.isfinite(base_grad_norm):
                base_grad_norm_values.append(base_grad_norm)

            if np.isfinite(transformer_grad_norm):
                transformer_grad_norm_values.append(transformer_grad_norm)

            if np.isfinite(correction_grad_norm):
                correction_grad_norm_values.append(correction_grad_norm)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
            )

            grad_norm_value = float(grad_norm.detach().cpu())

            if np.isfinite(grad_norm_value):
                grad_norm_values.append(grad_norm_value)

                if grad_norm_value > grad_clip_norm:
                    clipped_count += 1

            scaler.step(optimizer)
            scaler.update()

            # -----------------------------------------------------
            # Attention entropy
            #
            # During PCC-only epochs, the transformer is bypassed,
            # so attention should be None.
            # -----------------------------------------------------
            attn_entropy_value = None

            transformer_module = (
                model.module.transformer
                if hasattr(model, "module")
                else model.transformer
            )

            if hasattr(transformer_module, "last_attn_weights"):
                attn = transformer_module.last_attn_weights

                if attn is not None:
                    attn_entropy_value = attention_entropy(
                        attn.detach()
                    ).item()

            if attn_entropy_value is not None:
                entropy_sum += attn_entropy_value
                entropy_count += 1

            if epoch == 0 and step == 0:
                print(
                    "attention entropy:",
                    attn_entropy_value,
                    flush=True,
                )

            # -----------------------------------------------------
            # Accumulate training loss
            # -----------------------------------------------------
            if (
                hasattr(criterion, "reduction")
                and criterion.reduction == "mean"
            ):
                train_loss_sum += loss.detach() * x.size(0)
                n_seen += x.size(0)
            else:
                train_loss_sum += loss.detach()
                n_seen += 1

        if n_seen == 0:
            raise RuntimeError("No training samples were processed.")

        train_loss = (train_loss_sum / n_seen).item()

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------
        val_loss = float("nan")

        if val_loader is not None:
            model.eval()

            val_loss_sum = torch.zeros((), device=device)
            n_val = 0

            with torch.no_grad():
                for x_val, tgt_val in val_loader:
                    x_val = x_val.to(
                        device,
                        non_blocking=True,
                    )
                    tgt_val = tgt_val.to(
                        device,
                        non_blocking=True,
                    )

                    if dynamic_channels:
                        x_val = x_val[:, selected_idx_tensor, :]

                    channel_ids_val = selected_ids.unsqueeze(0).expand(
                        x_val.size(0),
                        -1,
                    )

                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=(device.type == "cuda"),
                    ):
                        pred_val = model(
                            x_val,
                            channel_ids=channel_ids_val,
                        ).squeeze(1)

                        loss_val = criterion(
                            pred_val,
                            tgt_val,
                        )

                    if (
                        hasattr(criterion, "reduction")
                        and criterion.reduction == "mean"
                    ):
                        val_loss_sum += (
                            loss_val.detach() * x_val.size(0)
                        )
                        n_val += x_val.size(0)
                    else:
                        val_loss_sum += loss_val.detach()
                        n_val += 1

            if n_val == 0:
                raise RuntimeError(
                    "No validation samples were processed."
                )

            val_loss = (val_loss_sum / n_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0

                # Optional best-checkpoint saving:
                #
                # model_to_save = (
                #     model.module
                #     if hasattr(model, "module")
                #     else model
                # )
                #
                # torch.save(
                #     model_to_save.state_dict(),
                #     os.path.join(
                #         logger.outdir,
                #         "best_model.pt",
                #     ),
                # )

            else:
                epochs_no_improve += 1

        # ---------------------------------------------------------
        # Epoch statistics
        # ---------------------------------------------------------
        mean_entropy = (
            entropy_sum / entropy_count
            if entropy_count > 0
            else float("nan")
        )

        if len(grad_norm_values) > 0:
            mean_grad_norm = np.mean(grad_norm_values)
            max_grad_norm = np.max(grad_norm_values)
            p95_grad_norm = np.percentile(grad_norm_values, 95)
            clip_fraction = clipped_count / len(grad_norm_values)
        else:
            mean_grad_norm = float("nan")
            max_grad_norm = float("nan")
            p95_grad_norm = float("nan")
            clip_fraction = float("nan")

        mean_base_grad_norm = (
            np.mean(base_grad_norm_values)
            if base_grad_norm_values 
            else float('nan')
        )

        mean_transformer_grad_norm = (
            np.mean(transformer_grad_norm_values)
            if transformer_grad_norm_values 
            else float('nan')
        )

        mean_correction_grad_norm = (
            np.mean(correction_grad_norm_values)
            if correction_grad_norm_values 
            else float('nan')
        )

        # ---------------------------------------------------------
        # Mean transformer/correction diagnostics across TRAIN batches
        # ---------------------------------------------------------
        if diagnostic_count > 0:
            mean_alpha = alpha_sum / diagnostic_count
            mean_gate = gate_sum / diagnostic_count
            mean_pcc_norm = pcc_norm_sum / diagnostic_count
            mean_correction_norm = correction_norm_sum / diagnostic_count
            mean_scaled_correction_norm = (
                scaled_correction_norm_sum / diagnostic_count
            )
            mean_raw_correction_ratio = (
                raw_correction_ratio_sum / diagnostic_count
            )
            mean_correction_ratio = (
                correction_ratio_sum / diagnostic_count
            )
        else:
            mean_alpha = float("nan")
            mean_gate = float("nan")
            mean_pcc_norm = float("nan")
            mean_correction_norm = float("nan")
            mean_scaled_correction_norm = float("nan")
            mean_raw_correction_ratio = float("nan")
            mean_correction_ratio = float("nan")

        # Store the LR used during this epoch, before scheduler.step().
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["attn_entropy"].append(mean_entropy)
        history["transformer_alpha"].append(mean_alpha)
        history["learning_rate"].append(current_lr)
        history["transformer_learning_rate"].append(transformer_lr)
        history["gradient_norm"].append(mean_grad_norm)
        history["base_gradient_norm"].append(mean_base_grad_norm)
        history["transformer_gradient_norm"].append(mean_transformer_grad_norm)
        history["correction_gradient_norm"].append(mean_correction_grad_norm)
        history["pcc_norm"].append(mean_pcc_norm)
        history["correction_norm"].append(mean_correction_norm)
        history["scaled_correction_norm"].append(
            mean_scaled_correction_norm
        )
        history["raw_correction_ratio"].append(
            mean_raw_correction_ratio
        )
        history["correction_ratio"].append(
            mean_correction_ratio
        )
        history["correction_gate"].append(mean_gate)

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

        logging.info(
            "Epoch %d/%d completed: "
            "train=%.6f, val=%.6f, "
            "alpha=%.4f, gate=%.4f, "
            "base_lr=%.3e, transformer_lr=%.3e, "
            "grad(mean=%.3f, p95=%.3f, max=%.3f, clipped=%.1f%%), "
            "attention_entropy=%s, "
            "pcc_norm=%.4f, raw_corr_norm=%.4f, "
            "scaled_corr_norm=%.4f, "
            "raw_corr/PCC=%.4f, applied_corr/PCC=%.4f",
            epoch + 1,
            max_epochs,
            train_loss,
            val_loss,
            mean_alpha,
            mean_gate,
            current_lr,
            transformer_lr,
            mean_grad_norm,
            p95_grad_norm,
            max_grad_norm,
            100.0 * clip_fraction,
            (
                f"{mean_entropy:.4f}"
                if np.isfinite(mean_entropy)
                else "N/A"
            ),
            mean_pcc_norm,
            mean_correction_norm,
            mean_scaled_correction_norm,
            mean_raw_correction_ratio,
            mean_correction_ratio,
        )

        # ---------------------------------------------------------
        # MultiStepLR is stepped once at the end of each epoch.
        #
        # Do not pass val_loss to MultiStepLR.
        # ---------------------------------------------------------
        if lr_scheduler is not None:
            lr_scheduler.step()

            new_lr = optimizer.param_groups[0]["lr"]
            new_transformer_lr = (
                optimizer.param_groups[1]["lr"]
                if len(optimizer.param_groups) > 1
                else new_lr
            )

            if (
                new_lr != current_lr
                or new_transformer_lr != transformer_lr
            ):
                logging.info(
                    "Learning rates changed after epoch %d: "
                    "base %.3e -> %.3e, transformer %.3e -> %.3e",
                    epoch + 1,
                    current_lr,
                    new_lr,
                    transformer_lr,
                    new_transformer_lr,
                )

        # ---------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------
        if (
            val_loader is not None
            and epochs_no_improve >= patience
        ):
            logging.info(
                "Early stopping at epoch %d. "
                "Best validation loss %.6f at epoch %d.",
                epoch + 1,
                best_val_loss,
                best_epoch + 1,
            )
            break

    if best_epoch >= 0:
        logging.info(
            "Best validation loss %.6f at epoch %d.",
            best_val_loss,
            best_epoch + 1,
        )
    else:
        logging.info(
            "No validation loss was recorded during training."
        )

    logging.info(
        "Training completed. Final train loss: %.6f",
        train_loss,
    )

    return history


def get_device(device):
    """Convenient function to set up hardware."""
    if device.lower() == "cpu":
        device = torch.device("cpu")
    elif device.lower() == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"-Use device: {device}")
        else:
            logging.warning("No MPS available. Use CPU instead.")
            device = torch.device("cpu")
    elif "cuda" in device.lower():
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            logging.warning("No GPU available. Use CPU instead.")
            device = torch.device("cpu")

    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory *= 1e-9
        logger.info("- Use device: {}".format(torch.cuda.get_device_name(device)))
        logger.info("device count: {}".format(torch.cuda.device_count()))
        logger.info("- Total memory: {:.4f} GB".format(total_memory))
    else:
        logger.info("- Use device: CPU")

    return device