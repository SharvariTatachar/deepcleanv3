import torch
import torch.nn as nn

class ConvBlock(nn.Module): 
    def __init__(self, cin: int, cout: int): 
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.Tanh()
    
    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))

class PerChannelDownsampler(nn.Module): 
    """
    Downsample each channel separately using DeepClean methodology.
    Input (B, C, L)
    Output (B, C, d, T)

    """

    def __init__(self, C: int, emb_dim: int): 
        super().__init__()
        self.C = C 
        self.emb_dim = emb_dim 
        self.channel_emb = nn.Embedding(C, emb_dim)

        # Conv blocks 
        self.block1 = ConvBlock(1, 8)
        self.block2 = ConvBlock(8, 16)
        self.block3 = ConvBlock(16, 32)
        self.block4 = ConvBlock(32, 64)
        self.block5 = ConvBlock(64, 128)
        self.block6 = ConvBlock(128, 128)
        self.block7 = ConvBlock(128, 128)

        # channel embedding projections 
        self.proj1 = nn.Linear(emb_dim, 8)
        self.proj2 = nn.Linear(emb_dim, 16)
        self.proj3 = nn.Linear(emb_dim, 32)
        self.proj4 = nn.Linear(emb_dim, 64)
        self.proj5 = nn.Linear(emb_dim, 128)
        self.proj6 = nn.Linear(emb_dim, 128)
        self.proj7 = nn.Linear(emb_dim, 128)
       
    def add_channel_emb(
        self, 
        y: torch.Tensor, 
        proj: nn.Linear, 
    ): 
        """
        y: (B, C, F, T)
        proj: projects base channel embedding -> F 
        """
        B, C, F, T = y.shape 
        channel_ids = torch.arange(C, device=y.device)
        e = self.channel_emb(channel_ids)
        e = proj(e)
        # Broadcast (C, F) -> (B, C, F, T)
        e = e.unsqueeze(0).unsqueeze(-1)
        return y + e 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        B, C, L = x.shape 

        # Block 1
        y = x.reshape(B * C, 1, L)
        y = self.block1(y)                                 # (B*C, 8, T1)
        y = y.reshape(B, C, 8, y.shape[-1])               # (B, C, 8, T1)
        y = self.add_channel_emb(y, self.proj1)

        # Block 2
        y = y.reshape(B * C, 8, y.shape[-1])
        y = self.block2(y)                                 # (B*C, 16, T2)
        y = y.reshape(B, C, 16, y.shape[-1])
        y = self.add_channel_emb(y, self.proj2)

        # Block 3
        y = y.reshape(B * C, 16, y.shape[-1])
        y = self.block3(y)                                 # (B*C, 32, T3)
        y = y.reshape(B, C, 32, y.shape[-1])
        y = self.add_channel_emb(y, self.proj3)

        # Block 4
        y = y.reshape(B * C, 32, y.shape[-1])
        y = self.block4(y)                                 # (B*C, 64, T4)
        y = y.reshape(B, C, 64, y.shape[-1])
        y = self.add_channel_emb(y, self.proj4)

        # Block 5
        y = y.reshape(B * C, 64, y.shape[-1])
        y = self.block5(y)                                 # (B*C, 128, T5)
        y = y.reshape(B, C, 128, y.shape[-1])
        y = self.add_channel_emb(y, self.proj5)

        # Block 6
        y = y.reshape(B * C, 128, y.shape[-1])
        y = self.block6(y)                                 # (B*C, 128, T6)
        y = y.reshape(B, C, 128, y.shape[-1])
        y = self.add_channel_emb(y, self.proj6)

        # Block 7
        y = y.reshape(B * C, 128, y.shape[-1])
        y = self.block7(y)                                 # (B*C, 128, T7)
        y = y.reshape(B, C, 128, y.shape[-1])
        y = self.add_channel_emb(y, self.proj7)

        return y


class Upsampler(nn.Module): 
    """
    Upsampling, using DeepClean net. 
    Input (B, F, Tds)   -- F token size, Tds downsampled time dim
    Output (B, 1, L)  -- L original time dimension 
    """
    def __init__(self):
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
        self.upsampler.add_module('CONVTRANS_5', nn.ConvTranspose1d(8, 8, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_5', nn.BatchNorm1d(8))
        self.upsampler.add_module('TANH_5', nn.Tanh())
        self.upsampler.add_module('CONVTRANS_6', nn.ConvTranspose1d(8, 8, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_6', nn.BatchNorm1d(8))
        self.upsampler.add_module('TANH_6', nn.Tanh())
        self.upsampler.add_module('CONVTRANS_7', nn.ConvTranspose1d(8, 1, kernel_size=7, stride=2, padding=3, output_padding=1))
        # self.upsampler.add_module('BN_5', nn.BatchNorm1d(1))
        # self.upsampler.add_module('TANH_5', nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.upsampler(x)
        return y 