import torch
import torch.nn as nn
import deepclean as dc 
import deepclean.timeseries as ts 

class WindowTokenizer(nn.Module): 
    """
    Encode one window (C, L) -> token (d)
    Linear over flattened window.
    """
    def __init__(self, C: int, L: int, d: int):
        super().__init__()
        self.C, self.L, self.d = C, L, d
        self.proj = nn.Linear(C * L, d)

    def forward(self, x_win: torch.Tensor) -> torch.Tensor:
        # x_win: (B, C, L)
        B, C, L = x_win.shape
        assert C == self.C and L == self.L
        return self.proj(x_win.reshape(B, C * L))  # (B, d)

class WindowDetokenizer(nn.Module): 
    """
    Decode token (d) to window (C, L)
    """
    def __init__(self, C: int, L: int, d: int):
        super().__init__()
        self.C, self.L, self.d = C, L, d
        self.proj = nn.Linear(d, C * L)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, d)
        B, d = z.shape
        assert d == self.d
        x = self.proj(z).reshape(B, self.C, self.L)
        return x 


class WindowTokenTransformer(nn.Module):
    """
    One token per 8s window (across all channels)
    Attention over windows (K tokens)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model = d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True, 
            activation ="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
      

    def forward(self, Z):
        """
        Z: (B, K, d)
        """
        return self.encoder(Z)
         