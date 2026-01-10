import torch
import torch.nn as nn
from . import windowtokentransformer as tt
from . import perchannelcnn as pcc
class HybridTransformerCNN(nn.Module):
    """
    Input: x (B, C, T_total) , T_total = K x L 
    Output: y (B, 1, T_total)
    """
    def __init__(self, C:int, fs: int, window_sec: float = 8.0, K: int = 4, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 1, cnn_kernel: int = 7, cnn_layers: int = 1):
        super().__init__()
        self.C = C 
        self.fs = fs 
        self.L = int(round(window_sec * fs)) # samples per window 
        self.K = K 


        self.tokenizer = tt.WindowTokenizer(C, self.L, d_model)
        self.transformer = tt.WindowTokenTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.detokenizer = tt.WindowDetokenizer(C, self.L, d_model)

        self.cnn = pcc.PerChannelCNN(C, kernel_size=cnn_kernel, n_layers=cnn_layers)
        
    
        self.readout = nn.Conv1d(C, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, T = x.shape 
        assert C == self.C 
        assert T == self.K * self.L, f"Expected T={self.K * self.L}, got {T}"
        
        x_win = x.reshape(B, C, self.K, self.L)
        x_win = x_win.permute(0, 2, 1,3).contiguous() 
        # print('after permute (B,K,C,L): ', x_win.shape)
        # Tokenize each window: (B, K, d)
        Z = torch.stack([self.tokenizer(x_win[:, k]) for k in range(self.K)], dim=1)

        # Attention over windows 
        Z = self.transformer(Z)

        # Detokenize to windows (B, K, C, L)
        x_win_rec = torch.stack([self.detokenizer(Z[:, k]) for k in range(self.K)], dim=1)
        x_rec = x_win_rec.permute(0,2,1,3).contiguous().reshape(B, C, T)

        # Per-channel CNN over time 
        x_rec = self.cnn(x_rec)
        y = self.readout(x_rec) # (B, 1, T_total)
        return y 
