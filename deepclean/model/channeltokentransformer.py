import torch
import torch.nn as nn

class ChannelTokenTransformer(nn.Module):
    """
    Tokens: channels (sequence length C)
    Attention operates over channels
    Input/Output: (B, C, 1)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 16,
        num_layers: int = 1,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0
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
        Z: (B, C, 1) 
        returns: (B, C, 1)
        """
        return self.encoder(Z)
         