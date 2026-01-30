import torch
import torch.nn as nn

class PerChannelDownsampler(nn.Module): 
    """
    Downsample each channel separately using DeepClean methodology.
    Input 
    Output  
    """

    def __init__(self, C: int, n_layers: int = 5, kernel_size: int = 2, stride: int = 2): 
        super().__init__()
        
        self.downsampler = nn.Sequential() 
        self.downsampler.add_module('CONV_1', nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_1', nn.BatchNorm(8))
        self.downsampler.add_module('TANH_1', nn.Tanh())
        self.downsampler.add_module('CONV_2', nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_2', nn.BatchNorm(16))
        self.downsampler.add_module('TANH_2', nn.Tanh())
        self.downsampler.add_module('CONV_3', nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_3', nn.BatchNorm(32))
        self.downsampler.add_module('TANH_3', nn.Tanh())
        self.downsampler.add_module('CONV_4', nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_4', nn.BatchNorm(64))
        self.downsampler.add_module('TANH_4', nn.Tanh())
        self.downsampler.add_module('CONV_5', nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_5', nn.BatchNorm(128))
        self.downsampler.add_module('TANH_5', nn.Tanh())
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        B, C, L = x.shape 
        x = x.reshape(B*C, 1, L)
        y = self.downsampler(x)
        y = y.reshape(B, C, 128, y.shape[-1])
        return y 


class Upsampler(nn.Module): 
    """
    Upsampling, using DeepClean net. 
    Input (B, d, T)   -- d transformer size, T downsampled time dim
    Output (B, 1, L)  -- L original time dimension 
    """
    def __init__(self, C: int, n_layers: int = 5, mode: str = "linear", 
                smooth_kernel: int = 7, target_len: int | None = None):
        super().__init__()
        
        self.upsampler = nn.Sequential() 
        self.upsampler.add_module('CONVTRANS_1', nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_1', nn.BatchNorm1d(64))
        self.upsampler.add_module('TANH_1', nn.Tanh())
        self.upsampler.add_module('CONVTRANS_2', nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_2', nn.BatchNorm1d(32))
        self.upsampler.add_module('TANH_2', nn.Tanh())
        self.upsampler.add_module('CONVTRANS_3', nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_3', nn.BatchNorm1d(16))
        self.upsampler.add_module('TANH_3', nn.Tanh())
        self.upsampler.add_module('CONVTRANS_4', nn.ConvTranspose1d(16, 8, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_4', nn.BatchNorm1d(8))
        self.upsampler.add_module('TANH_4', nn.Tanh())
        self.upsampler.add_module('CONVTRANS_4', nn.ConvTranspose1d(8, 1, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_4', nn.BatchNorm1d(1))
        self.upsampler.add_module('TANH_4', nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape 
        y = self.upsampler(x)
        return y 