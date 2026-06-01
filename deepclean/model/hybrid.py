import sys
import torch
import torch.nn as nn
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
    Input: x (B, C, L) , L = 8s * fs 
    Output: y (B, 1, L)
    """
    def __init__(self, C:int, fs: int, window_sec: float = 8.0, d_model: int = 128,
                 nhead: int = 16, num_layers: int = 2, cnn_kernel: int = 2, cnn_layers: int = 5, 
                 n_iters: int = 2, num_channel_ids: int = None):
        super().__init__()
        self.n_iters = n_iters 
        self.C = C 
        self.fs = fs 
        self.L = int(round(window_sec * fs)) 
        self.d_model = d_model
        
        self.downsample = pcc.PerChannelDownsampler(self.C, emb_dim = self.d_model)
       
        self.channel_embedding = nn.Embedding(
            num_embeddings=num_channel_ids,
            embedding_dim=d_model,
        )
       
        self.transformer = tt.ChannelTokenTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)
       
        self.upsample = pcc.Upsampler()
       

    def forward(self, x: torch.Tensor, channel_ids=None) -> torch.Tensor:
        # Per channel downsampler 
        B, C, T = x.shape
        #print('input shape x:', x.shape)
        # perm = torch.randperm(C, device=x.device)
        # inv_perm = torch.argsort(perm)
        # x_perm = x[:, perm, :]
        
        ch_emb = self.channel_embedding(channel_ids)
        # ch_emb_perm = ch_emb[:, perm, :]

        x_ds = self.downsample(x, ch_emb)
        # x_ds2 = self.downsample(x_perm, ch_emb_perm)

        # x_ds2_unperm = x_ds2[:, inv_perm, :, :]

        # diff = (x_ds - x_ds2_unperm).abs().mean()
        # print('downsample diff', diff, flush=True)
    
        # Reshape, so each timestep gets passed to transformer 
        B, C, F, Tds = x_ds.shape 
        y_bt = x_ds.permute(0,3,1,2).contiguous().view(B*Tds, C, F)

        if channel_ids is None: 
            raise ValueError('channel_ids must be passed for name-based channel embeddings')

        if channel_ids.dim() == 1: 
            channel_ids = channel_ids.unsqueeze(0).expand(B, -1)

        channel_ids = channel_ids.to(x.device)
        assert channel_ids.max().item() < self.channel_embedding.num_embeddings, (
            f"channel id {channel_ids.max().item()} out of range for "
            f"{self.channel_embedding.num_embeddings} embeddings"
        )

        ch_emb = self.channel_embedding(channel_ids)
        # ch_emb: (B, C, F)
        ch_emb_bt = (
        ch_emb
        .unsqueeze(1)                 # (B, 1, C, F)
        .expand(B, Tds, C, F)          # (B, Tds, C, F)
        .contiguous()
        .view(B * Tds, C, F)           # (B*Tds, C, F)
        )
    
        y_bt = y_bt + ch_emb_bt


        self.last_selected_channel_ids = channel_ids.detach().cpu()
        self.last_y_bt = y_bt.detach().cpu()
        # print('transformer input: ', y_bt.shape)
        z_bt= self.transformer(y_bt) 
        self.last_z_bt = z_bt.detach().cpu()      # after transformer
        self.last_B = B
        self.last_C = C
        self.last_Tds = Tds
        z = z_bt.view(B, Tds, C, F).permute(0, 2, 3, 1).contiguous()
        
        # Mean-pooling 
        z_pooled = z.mean(dim=1) # (B, F, T')

        # Upsampler 
        y = self.upsample(z_pooled) # (B, 1, L)

        return y  

        


