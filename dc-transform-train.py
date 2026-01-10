import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 

import deepclean as dc 
import deepclean.timeseries as ts 
import deepclean.model as model 
import deepclean.model.hybrid as hy
import deepclean.model.utils as utils

device = utils.get_device('mps')
train_data = ts.TimeSeriesSegmentDataset(kernel=8, stride=8.0, pad_mode='median')


train_data.read('combined_data.npz', channels='channels.ini',
    start_time=1378403243, end_time=1378403243+3072, fs=2048)  
train_data = train_data.bandpass(110, 130, order=8, channels='target')

# filter pad default from deepclean-prod, is 5: 
filt_pad = 5 
fs = 2048 # from info.txt 
train_data.data = train_data.data[:, int(filt_pad * fs):-int(filt_pad * fs)]

mean = train_data.mean 
std = train_data.std 
train_data = train_data.normalize(mean, std)

# aux_patch, tgt_patch = train_data[0]
# print(aux_patch.shape, tgt_patch.shape)

K = 4 # change to 4 later and give longer data segment/diff time unit 
kw_ds = ts.KWindowDataset(train_data, K=K)
loader = DataLoader(kw_ds, batch_size=4, shuffle=False, num_workers=0)

# x, tgt = next(iter(loader))
# print('x: ', x.shape)  # (B, C, K*L) 
# print('tgt: ', tgt.shape) # (B, K*L)

x, tgt = next(iter(loader))

model = hy.HybridTransformerCNN(C=x.shape[1], fs=2048, window_sec=8.0, K=K,
                                       d_model=128, nhead=4, num_layers=1,
                                       cnn_kernel=7, cnn_layers=1)

# print('expected: (B,C, K*L)=', (x.shape[0], x.shape[1], K * model.L))

criterion = nn.MSELoss() # TODO: add PSD loss  
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

model.train() 
# with torch.no_grad(): 
#     pred = model(x)

# print("x: ", x.shape)
# print("pred: ", pred.shape)
