import torch
import torch.nn as nn


class ChannelTokenizer(nn.Module): 
    """
    Input: x (B, C, L)
    Output: z (B, C, d)
    Encode one token length L (8s x 2048 Hz) to a summary d
    """
    def __init__(self, C: int, L: int, d: int):
        super().__init__()
        self.C, self.L, self.d = C, L, d
        self.proj = nn.Linear(L, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        B, C, L = x.shape
        assert C == self.C and L == self.L
        x2 = x.reshape(B*C, L) # to apply linear layer to each channel separately. 
        z2 = self.proj(x2) # (B * C, d), TODO: replace with something lighter 
        Z = z2.reshape(B, C, self.d)
        return Z

class ChannelTokenTransformer(nn.Module):
    """
    Tokens: channels (sequence length C)
    Attention operates over channels
    Input/Output: (B, C, d)
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
        Z: (B, C, d)
        """
        return self.encoder(Z)
         
    
# Temporary wrapper class to check functionality: 

# class ChannelAttentionBlock(nn.Module): 
#     """
#     x (B,C,L) -> Z' (B,C,d)
#     """
#     def __init__(self, C:int, L:int, d_model: int = 128, nhead: int = 4, num_layers: int = 1):
#         super().__init__()
#         self.tokenizer = ChannelTokenizer(C=C, L=L, d=d_model)
#         self.transformer = ChannelTokenTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         Z = self.tokenizer(x)          # (B,C,d)
#         Z = self.transformer(Z)        # (B,C,d)
#         return Z

# B, C, L = 4, 35, 16384
# x = torch.randn(B, C, L)
# blk = ChannelAttentionBlock(C=C, L=L, d_model=128)
# Z = blk(x)
# print(Z.shape)