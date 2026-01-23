import torch
import torch.nn as nn
from . import channeltokentransformer as tt
from . import perchannelcnn as pcc
class HybridTransformerCNN(nn.Module):
    """
    Input: x (B, C, L) , L = 8s * fs 
    Output: y (B, 1, L)
    """
    def __init__(self, C:int, fs: int, window_sec: float = 8.0, d_model: int = 64,
                 nhead: int = 2, num_layers: int = 1, cnn_kernel: int = 3, cnn_layers: int = 1):
        super().__init__()
        self.C = C 
        self.fs = fs 
        self.L = int(round(window_sec * fs)) 
        
        # tokens are channels 
        self.tokenizer = tt.ChannelTokenizer(C=self.C, L=self.L, d=d_model)
        self.transformer = tt.ChannelTokenTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)

        self.cnn = pcc.PerChannelCNN(C, kernel_size=cnn_kernel, n_layers=cnn_layers)
        
        self.readout = nn.Conv1d(C, 1, kernel_size=1)
        
        # Projection layer to map from token dimension d back to time dimension L
        self.projection = nn.Linear(d_model, self.L) # TODO: replace? 

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, L = x.shape 
        assert C == self.C and L == self.L 
        
        Z = self.tokenizer(x) # (B, C, d)
        # Attention over channels 
        Z = self.transformer(Z) # (B, C, d)

        # Pass transformer output directly to CNN
        # CNN operates over token dimension d 
        Z_cnn = self.cnn(Z) # (B, C, d)
        
        y = self.readout(Z_cnn) # (B, 1, d)  # TODO: this linear mapping will break permutation invariance --> instead do a sum pooling or average, but you need to make sure to have enough transformer layers
        
        # Project from token dimension d back to original time dimension L
        y = y.squeeze(1)  # (B, d)
        y = self.projection(y)  # (B, L) 
        y = y.unsqueeze(1)  # (B, 1, L)
        
        return y 
