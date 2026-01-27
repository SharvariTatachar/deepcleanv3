import torch
import torch.nn as nn

class Downsampler(nn.Module): 
    """
    Downsample time using AvgPool1d layers
    Input: x (B, C, L)
    Output: y (B, C, d)
    """

    def __init__(self, n_layers: int = 5, kernel_size: int = 2, stride: int = 2): 
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
            for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        B, C, L = x.shape
        y = x.reshape(B * C, 1, L) # pooling per channel (separately)
        for pool in self.pools: 
            y = pool(y)
        L_out = y.shape[-1]
        y = y.reshape(B, C, L_out)
        return y 


class Upsampler(nn.Module): 
    """
    Upsample back and smooth using Conv layer 
    Input: y (B, 1, L_ds)
    Output: y_up (B, 1, L)
    """
    def __init__(self, n_layers: int = 5, mode: str = "linear"):
        super().__init__()
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
                                  for _ in range(n_layers)])
        self.smooth = nn.Conv1d(1, 1, kernel_size=7, padding=3)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        for up in self.ups:
            y = up(y)
        return self.smooth(y)
