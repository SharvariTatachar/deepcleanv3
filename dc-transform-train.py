import os
import torch 
import logging
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader 

import deepclean as dc 
import deepclean.timeseries as ts 
import deepclean.criterion
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
train_data = ts.TimeSeriesSegmentDataset(kernel=8, stride=0.25, pad_mode='median')
val_data = ts.TimeSeriesSegmentDataset(kernel=8, stride=0.25, pad_mode='median')
# TODO: overlap for validation set might have to be less than for training

t0 = 1378403243 

# not using the full 3072s 
train_data.read('combined_data.npz', channels='channels.ini',
    start_time=t0, end_time=t0+1024, fs=2048)  

val_data.read('combined_data.npz', channels='channels.ini',
    start_time=t0+1024, end_time=t0+2048, fs=2048) 

# print('train windows: ', len(train_data))
# print('val windows: ', len(val_data))
# test_data.read('compined_data.npz', channels='channels.ini', 
#     start_time=t0+2560, end_time=t0+3072, fs=2048)

train_data = train_data.bandpass(110, 130, order=8, channels='target')
val_data = val_data.bandpass(110, 130, order=8, channels='target')
# test_data = test_data.bandpass(110, 130, order=8, channels='target')


# filter pad default from deepclean-prod, is 5: 
filt_pad = 5 
fs = 2048 # from info.txt 
train_data.data = train_data.data[:, int(filt_pad * fs):-int(filt_pad * fs)]
val_data.data = val_data.data[:, int(filt_pad * fs):-int(filt_pad * fs)]
# test_data.data = test_data.data[:, int(filt_pad * fs):-int(filt_pad * fs)]

mean = train_data.mean 
std = train_data.std 
train_data = train_data.normalize()
val_data = val_data.normalize(mean, std)
# test_data = test_data.normalize(mean, std)

aux_patch, tgt_patch = train_data[0]
# print(aux_patch.shape, tgt_patch.shape)
single_train = ts.SingleDataset(train_data, fixed_id=0)
train_loader = DataLoader(single_train, batch_size=1, shuffle=False, num_workers=0)
single_val = ts.SingleDataset(val_data, fixed_id=0)
val_loader = DataLoader(single_val, batch_size=1, shuffle=False, num_worker=0)
# train_loader = DataLoader(train_data, batch_size=4, shuffle=False, num_workers=0)
# val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0)
x, tgt = next(iter(train_loader))
# print('x: ', x.shape)  # (B, C, L) 
# print('tgt: ', tgt.shape) # (B, L)

model = hy.HybridTransformerCNN(C=x.shape[1], fs=2048, window_sec=8.0,
                                       d_model=512, nhead=16, num_layers=1,
                                       cnn_kernel=2, cnn_layers=5)
model = model.to(device)

criterion = nn.MSELoss() 

optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-3)
lr_scheduler = None
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

train_logger = dc.logger.Logger(outdir=train_dir, metrics=['loss'])
utils.train(
    train_loader, model, criterion, device, optimizer, lr_scheduler, 
    val_loader=val_loader, max_epochs=5, logger=train_logger)


# with torch.no_grad():
    # pred = model(x)
    # print('pred shape: ', pred.shape)
   