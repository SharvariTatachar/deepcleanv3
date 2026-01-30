import torch
import torch.nn as nn
from . import channeltokentransformer as tt
from . import perchannelcnn as pcc
class HybridTransformerCNN(nn.Module):
    """
    Input: x (B, C, L) , L = 8s * fs 
    Output: y (B, 1, L)
    """
    def __init__(self, C:int, fs: int, window_sec: float = 8.0, d_model: int = 512,
                 nhead: int = 16, num_layers: int = 1, cnn_kernel: int = 2, cnn_layers: int = 5):
        super().__init__()
        self.C = C 
        self.fs = fs 
        self.L = int(round(window_sec * fs)) 
        self.d_model = d_model
        
        self.downsample = pcc.PerChannelDownsampler(self.C)
        self.transformer = tt.ChannelTokenTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.upsample = pcc.Upsampler()
       

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, L = x.shape 
        assert C == self.C and L == self.L 

        # Per-channel downsampler  
        x_ds = self.downsample(x)  # (B, C, F, Tds)
        # print("downsampler: ", x_ds.shape)

        # Reshaping, each timestep gets passed to transformer: 
        B, C, F, Tds = x_ds.shape 
        y_bt = x_ds.permute(0,3,1,2).contiguous().view(B*Tds, C, F)

        # print('transformer input: ', y_bt.shape)
        z_bt= self.transformer(y_bt) 
        z = z_bt.view(B, Tds, C, F).permute(0, 2, 3, 1).contiguous()
        # print("back to grid: ", z.shape)

        # Sum pooling over transformer output 
        z_pooled = z.sum(dim=1) # (B, F, Tds)

        # Upsampler 
        y = self.upsample(z_pooled) # (B, 1, L)

        return y 
