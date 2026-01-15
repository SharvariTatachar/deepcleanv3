import torch
import torch.nn as nn
from . import channeltokentransformer as tt
from . import perchannelcnn as pcc
class HybridTransformerCNN(nn.Module):
    """
    Input: x (B, C, L) , L = 8s * fs 
    Output: y (B, 1, L)
    """
    def __init__(self, C:int, fs: int, window_sec: float = 8.0, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 1, cnn_kernel: int = 7, cnn_layers: int = 1):
        super().__init__()
        self.C = C 
        self.fs = fs 
        self.L = int(round(window_sec * fs)) 
        
        # tokens are channels 
        self.tokenizer = tt.ChannelTokenizer(C=self.C, L=self.L, d_model)
        self.transformer = tt.ChannelTokenTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)

        self.cnn = pcc.PerChannelCNN(C, kernel_size=cnn_kernel, n_layers=cnn_layers)
        
    
        self.readout = nn.Conv1d(C, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, L = x.shape 
        assert C == self.C and L == self.L 
        
        Z = self.tokenizer(x) # (B, C, d)
        # Attention over channels 
        Z = self.transformer(Z) # (B, C, d)

        # TODO: change the CNN to convolutions over each window, per channel. 
        x_rec = self.cnn(x_rec)
        y = self.readout(x_rec) # (B, 1, T_total)
        return y 
