import os
import torch 
import logging
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader 

import deepclean as dc 
import deepclean.timeseries as ts 
import deepclean.model as model 
import deepclean.model.hybrid as hy
import deepclean.model.utils as utils

train_dir = 'train_dir'
os.makedirs(train_dir, exist_ok=True)
log = os.path.join(train_dir, 'log.log')

# set up logging to both file and console 
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log, mode='a'),  # Write to file
        logging.StreamHandler()  # Write to console
    ]
)
logging.info('Create training directory: {}'.format(train_dir))


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

#TODO: change the way your're building the data samples 
K = 4 
kw_ds = ts.KWindowDataset(train_data, K=K)
train_loader = DataLoader(kw_ds, batch_size=4, shuffle=False, num_workers=0)

x, tgt = next(iter(train_loader))
# print('x: ', x.shape)  # (B, C, K*L) 
# print('tgt: ', tgt.shape) # (B, K*L)

model = hy.HybridTransformerCNN(C=x.shape[1], fs=2048, window_sec=8.0, K=K,
                                       d_model=128, nhead=4, num_layers=1,
                                       cnn_kernel=7, cnn_layers=1)

# print('expected: (B,C, K*L)=', (x.shape[0], x.shape[1], K * model.L))
model = model.to(device)

criterion = nn.MSELoss() # TODO: change to composite PSD loss  
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

train_logger = dc.logger.Logger(outdir=train_dir, metrics=['loss'])
utils.train(
    train_loader, model, criterion, device, optimizer, lr_scheduler, 
    max_epochs=1, logger=train_logger)

# with torch.no_grad(): 
#     pred = model(x)

# print("x: ", x.shape)
# print("pred: ", pred.shape)

with torch.no_grad():
    zero_loss = torch.mean(tgt**2).item()
    print("zero baseline:", zero_loss)