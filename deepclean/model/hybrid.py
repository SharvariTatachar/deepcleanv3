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
        
        self.downsample = pcc.Downsampler(cnn_layers)
       
        self.transformer = tt.ChannelTokenTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)
        
        self.upsample = pcc.Upsampler(cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, L = x.shape 
        assert C == self.C and L == self.L 

        # Downsampler 
        x_ds = self.downsample(x)  # (B, C, d)
        # print("downsampler: ", x_ds.shape)

        # Attention over channels 
        Z = self.transformer(x_ds) # (B, C, d)
        # print("transformer: ", Z.shape)
        
        # Sum-poolreadout (preserves permutation invariance over channels)
        # Or mean-pool
        y_ds = Z.sum(dim=1, keepdim=True) # (B, 1, d) 

        # Upsampler 
        y = self.upsample(y_ds) 
        # print('upsampler: ', y.shape)
        return y 
