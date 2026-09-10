import sys
import torch
import torch.nn as nn
import math
import logging 
from . import channeltokentransformer as tt
from . import perchannelcnn as pcc

debug_logger = logging.getLogger("debug_logger")
debug_logger.setLevel(logging.INFO)

debug_handler = logging.FileHandler("debug_shapes.log", mode="w")
debug_handler.setLevel(logging.INFO)

debug_formatter = logging.Formatter(
    "%(asctime)s - %(message)s"
)

debug_handler.setFormatter(debug_formatter)

debug_logger.addHandler(debug_handler)


class HybridTransformerCNN(nn.Module):
    """
    Input:
        x: (B, C, L)

    Output:
        y: (B, 1, L)

    Training schedule:
        Epochs 0-4:
            Pure per-channel CNN path, transformer alpha = 0.

        Epochs 5-9:
            Gradually introduce transformer, alpha = 0.2 -> 1.0.

        Epoch 10 onward:
            Full transformer, alpha = 1.
    """

    def __init__(
        self,
        C: int,
        fs: int,
        window_sec: float = 8.0,
        d_model: int = 128,
        nhead: int = 16,
        num_layers: int = 2,
        cnn_kernel: int = 2,
        cnn_layers: int = 5,
        n_iters: int = 2,
        num_channel_ids: int = None,
        transformer_warmup_epochs: int = 5,
        transformer_ramp_epochs: int = 10,
        transformer_warmup_alpha: float = 0.01, 
        target_correction_ratio: float = 0.25,
        correction_head_freeze_epochs: int = 0, 
    ):
        super().__init__()

        if num_channel_ids is None:
            raise ValueError("num_channel_ids must be provided")

        self.n_iters = n_iters
        self.C = C
        self.fs = fs
        self.L = int(round(window_sec * fs))
        self.d_model = d_model

        # Epoch/ramp configuration
        self.current_epoch = 0
        self.transformer_warmup_epochs = transformer_warmup_epochs
        self.transformer_ramp_epochs = transformer_ramp_epochs
        self.transformer_warmup_alpha = transformer_warmup_alpha
        self.target_correction_ratio = target_correction_ratio
        self.downsample = pcc.PerChannelDownsampler(
            self.C,
            emb_dim=self.d_model,
        )

        self.channel_embedding = nn.Embedding(
            num_embeddings=num_channel_ids,
            embedding_dim=d_model,
        )

        self.transformer = tt.ChannelTokenTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )

        self.upsample = pcc.Upsampler()


        # Debugging values from the most recent forward pass
        self.last_selected_channel_ids = None
        self.last_y_bt = None
        self.last_z_bt = None
        self.last_transformer_alpha = 0.0
        self.last_B = None
        self.last_C = None
        self.last_Tds = None
        self.last_pcc_bt = None 
        self.last_correction_bt = None
        self.last_pcc_norm = float("nan")
        self.last_raw_correction_norm = float("nan")
        self.last_scaled_correction_norm = float("nan")
        self.last_applied_correction_norm = float("nan")
        self.last_raw_correction_ratio = float("nan")
        self.last_applied_correction_ratio = float("nan")
        self.correction_head = nn.Linear(
            self.d_model,
            self.d_model,
        )


        nn.init.normal_(
            self.correction_head.weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.zeros_(
            self.correction_head.bias
        )

        self.correction_head_freeze_epochs = correction_head_freeze_epochs
        self.correction_gate_logit = nn.Parameter(
            torch.tensor(-1.3863),
        )
        self.last_correction_gate = None 
        # self.correction_norm = nn.LayerNorm(self.d_model)

        # nn.init.normal_(
        #     self.correction_head.weight, 
        #     mean=0.0, 
        #     std=1e-3,
        # )

        # nn.init.zeros_(self.correction_head.bias)

    def set_epoch(self, epoch: int) -> None:
        """
        Called by the training loop at the start of each epoch.

        PyTorch epoch indices normally start at zero:
            epoch 0 = first epoch
            epoch 1 = second epoch
            ...
        """
        self.current_epoch = int(epoch)

    def set_correction_head_trainable(self, trainable:bool) -> None:
        """ Freeze or unfreeze the transformer correction head"""
        for param in self.correction_head.parameters():
            param.requires_grad = trainable

    def get_transformer_alpha(self) -> float:
        """
        Correction-branch coefficient.

        During warmup, use a small nonzero coefficient so the transformer
        and correction head receive gradients while having very little
        influence on the PCC representation.

        After warmup, smoothly ramp from warmup_alpha to 1.0.
        """
        epoch = self.current_epoch
        alpha_min = self.transformer_warmup_alpha

        if epoch < self.transformer_warmup_epochs:
            return alpha_min

        if self.transformer_ramp_epochs <= 0:
            return 1.0

        progress = (
            epoch - self.transformer_warmup_epochs + 1
        ) / self.transformer_ramp_epochs

        progress = min(max(progress, 0.0), 1.0)

        cosine_progress = 0.5 * (
            1.0 - math.cos(math.pi * progress)
        )

        # Smoothly move from alpha_min to 1.0.
        alpha = alpha_min + (1.0 - alpha_min) * cosine_progress

        return alpha

    def forward(
        self,
        x: torch.Tensor,
        channel_ids=None,
    ) -> torch.Tensor:

        if channel_ids is None:
            raise ValueError(
                "channel_ids must be passed for name-based channel embeddings"
            )

        B, C, _ = x.shape

        if channel_ids.dim() == 1:
            channel_ids = channel_ids.unsqueeze(0).expand(B, -1)

        channel_ids = channel_ids.to(
            device=x.device,
            dtype=torch.long,
        )

        if channel_ids.shape != (B, C):
            raise ValueError(
                "channel_ids must have shape (B, C). "
                f"Received {tuple(channel_ids.shape)} for input "
                f"shape {tuple(x.shape)}."
            )

        max_channel_id = channel_ids.max().item()

        if max_channel_id >= self.channel_embedding.num_embeddings:
            raise ValueError(
                f"Channel ID {max_channel_id} is out of range for "
                f"{self.channel_embedding.num_embeddings} embeddings."
            )

        # ---------------------------------------------------------
        # Channel embeddings
        # ---------------------------------------------------------
        ch_emb = self.channel_embedding(channel_ids)  # (B, C, F)

        # ---------------------------------------------------------
        # Per-channel CNN downsampler
        # ---------------------------------------------------------
        # Expected output shape: (B, C, F, Tds)
        #
        # Because ch_emb is passed into the downsampler, the PCC
        # representation is already conditioned on channel identity.
        x_ds = self.downsample(x, ch_emb)

        B, C, F, Tds = x_ds.shape

        # Reshape PCC features so that each downsampled time step is
        # processed as a separate channel-token sequence:
        #
        # (B, C, F, Tds) -> (B*Tds, C, F)
        pcc_bt = (
            x_ds
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(B * Tds, C, F)
        )

        alpha = self.get_transformer_alpha()
        self.last_transformer_alpha = float(alpha)

        # ---------------------------------------------------------
        # Transformer correction path
        # ---------------------------------------------------------
        # Expand channel embeddings across downsampled time steps:
        #
        # (B, C, F) -> (B*Tds, C, F)
        ch_emb_bt = (
            ch_emb
            .unsqueeze(1)                 # (B, 1, C, F)
            .expand(B, Tds, C, F)         # (B, Tds, C, F)
            .contiguous()
            .view(B * Tds, C, F)          # (B*Tds, C, F)
        )

        # The PCC downsampler already receives channel embeddings.
        # Adding them explicitly here gives channel identity directly
        # to the cross-channel transformer.
        transformer_input = pcc_bt + ch_emb_bt
        transformer_features = self.transformer(transformer_input)
        correction_bt = self.correction_head(transformer_features)
        # ---------------------------------------------------------
        # Scale correction relative to PCC feature magnitude
        # ---------------------------------------------------------
        target_correction_ratio = self.target_correction_ratio
        eps = 1e-6

        # Per channel-token RMS across the latent feature dimension.
        # Detach the norms so this scaling acts as a magnitude constraint
        # rather than another gradient path.
        pcc_rms = (
            pcc_bt.detach()
            .pow(2)
            .mean(dim=-1, keepdim=True)
            .sqrt()
        )

        correction_rms = (
            correction_bt.detach()
            .pow(2)
            .mean(dim=-1, keepdim=True)
            .sqrt()
        )

        correction_scale = (
            target_correction_ratio
            * pcc_rms
            / (correction_rms + eps)
        )

        # Avoid extreme amplification when the correction is initially tiny.
        correction_scale = correction_scale.clamp(max=5.0)

        scaled_correction_bt = correction_bt * correction_scale
        gate = torch.sigmoid(self.correction_gate_logit)
        self.last_correction_gate = gate.detach().item()
        #mixed_bt = pcc_bt + alpha * scaled_correction_bt
       
        mixed_bt = (
            pcc_bt
            + alpha * gate * scaled_correction_bt
        )
        # correction_bt = self.correction_norm(correction_bt)
        # mixed_bt = pcc_bt + alpha * correction_bt

        with torch.no_grad():
            base_norm = pcc_bt.norm(dim=-1).mean()
            raw_correction_norm = correction_bt.norm(dim=-1).mean()
            scaled_correction_norm = scaled_correction_bt.norm(dim=-1).mean()
            applied_correction_norm = (
                alpha * gate * scaled_correction_bt
            ).norm(dim=-1).mean()

            self.last_pcc_norm = base_norm.item()
            self.last_raw_correction_norm = raw_correction_norm.item()
            self.last_scaled_correction_norm = scaled_correction_norm.item()
            self.last_applied_correction_norm = applied_correction_norm.item()

            self.last_raw_correction_ratio = (
                raw_correction_norm / (base_norm + 1e-8)
            ).item()

            self.last_applied_correction_ratio = (
                applied_correction_norm / (base_norm + 1e-8)
            ).item()

        self.last_selected_channel_ids = channel_ids.detach().cpu()
        self.last_B = B
        self.last_C = C
        self.last_Tds = Tds

        # ---------------------------------------------------------
        # Restore convolutional feature shape
        # ---------------------------------------------------------
        # (B*Tds, C, F) -> (B, C, F, Tds)
        z = (
            mixed_bt
            .view(B, Tds, C, F)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        # ---------------------------------------------------------
        # Pool across witness channels
        # ---------------------------------------------------------
        z_pooled = z.mean(dim=1)  # (B, F, Tds)

        # ---------------------------------------------------------
        # Upsample back to the original time resolution
        # ---------------------------------------------------------
        y = self.upsample(z_pooled)  # expected: (B, 1, L)

        return y