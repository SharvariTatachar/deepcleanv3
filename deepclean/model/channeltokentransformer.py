import torch
import torch.nn as nn

class AttnEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        self.last_attn = None

    def forward(self, x):
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False,
        )

        # attn_weights shape: (B, num_heads, C, C)
        self.last_attn = attn_weights.detach()

        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))

        return x


class ChannelTokenTransformer(nn.Module):
    """
    Tokens: channels (sequence length C)
    Attention operates over channels
    Input/Output: (B, C, 1)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert d_model % nhead == 0
        # layer = nn.TransformerEncoderLayer(
        #     d_model = d_model, nhead=nhead, 
        #     dim_feedforward=dim_feedforward, 
        #     dropout=dropout, 
        #     batch_first=True, 
        #     activation ="gelu"
        # )
        # self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.layers = nn.ModuleList([
            AttnEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
      

    def forward(self, Z):
        """
        Z: (B, C, 1) 
        returns: (B, C, 1)
        """
        #return self.encoder(Z)
        for layer in self.layers: 
            z=layer(Z)
        return z 
    
    def get_attention(self): 
        return[layer.last_attn for layer in self.layers]
         