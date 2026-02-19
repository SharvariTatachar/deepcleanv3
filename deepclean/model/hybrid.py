import torch
import torch.nn as nn
from . import channeltokentransformer as tt
from . import perchannelcnn as pcc
class HybridTransformerCNN(nn.Module):
    """
    Input: x (B, C, L) , L = 8s * fs 
    Output: y (B, 1, L)
    """
    def __init__(self, C:int, fs: int, window_sec: float = 8.0, d_model: int = 512,
                 nhead: int = 16, num_layers: int = 2, cnn_kernel: int = 2, cnn_layers: int = 5, n_iters: int = 2):
        super().__init__()
        self.n_iters = n_iters 
        self.C = C 
        self.fs = fs 
        self.L = int(round(window_sec * fs)) 
        self.d_model = d_model
        
        self.downsample = pcc.PerChannelDownsampler(self.C)
        self.transformer = tt.ChannelTokenTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.upsample = pcc.Upsampler()
       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Iterative block 
            for it < n_iters-1: downsample -> transformer -> upsample, repeat
            for it == n_iters-1: downsample -> transformer -> sum pooling -> upsample 
        """
        # x_curr = x # (B, C, L)
        # for it in range(self.n_iters): 
        #     B, C, L = x_curr.shape 
        #     assert C == self.C and L == self.L 

        #     # Per-channel downsampler  
        #     x_ds = self.downsample(x_curr)  # (B, C, F, Tds)
        #     # print("downsampler: ", x_ds.shape)

        #     # Reshaping, each timestep gets passed to transformer: 
        #     B, C, F, Tds = x_ds.shape 
        #     y_bt = x_ds.permute(0,3,1,2).contiguous().view(B*Tds, C, F)

        #     # print('transformer input: ', y_bt.shape)
        #     z_bt= self.transformer(y_bt) 
        #     z = z_bt.view(B, Tds, C, F).permute(0, 2, 3, 1).contiguous()
        #     # print("back to grid: ", z.shape)
        #     if it < self.n_iters-1: 
        #         z_pc = z.view(B*C, F, Tds) 
        #         y_pc = self.upsample(z_pc)
        #         x_curr = y_pc.view(B, C, L) # feed to next iter
            
        #     # Final iteration 
        #     # Sum pooling over transformer output 
        #     else: 
        #         z_pooled = z.sum(dim=1) # (B, F, Tds)
        #         y = self.upsample(z_pooled) # (B, 1, L)

        # return y 

        # Per channel downsampler 
        x_ds = self.downsample(x)
        print('downsampler: ', x_ds.shape) # (B, F, Tds)
        y = self.upsample(x_ds) # (B, 1, L)
        return y  

        


