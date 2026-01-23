import torch
import torch.nn as nn
class PerChannelCNN(nn.Module):
    """
    Depthwise 1D CNN over time.

    - Processes each channel independently (groups = C).
    - No mixing between channels inside this block.

    Input:  x of shape (batch, C, T)
    Output: y of shape (batch, C, T_out)  (T_out can equal T if stride=1)
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        n_layers: int = 1,
    ):
        super().__init__()
        padding = (kernel_size -1) // 2 

        layers = []
        in_channels = num_channels
        out_channels = num_channels

        for _ in range(n_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=num_channels,  # depthwise: independent per channel
                    bias=True,
                )
            )
            layers.append(nn.BatchNorm1d(num_channels))
            layers.append(nn.Tanh())  # or nn.ReLU(), etc.

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch, C, T)
        returns: (batch, C, T_out)
        """
        # B, C, T = x.shape
        # assert C == self.num_channels, "Channel dim mismatch"
        return self.net(x)