import os
import torch 
import pickle
import argparse
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
import deepclean.model.deepclean
import configs.110config as config

def parse_cmd():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), usage='%(prog)s [options]')
    
    # dataset arguments 
    parser.add_argument('--train-t0', help='GPS of the first sample', type=int)
    parser.add_argument('--train-duration', help='Duration of train/val frame', type=int)
    parser.add_argument('--chanslist', help='Path to channel list', type=str)
    parser.add_argument('--fs', help='Sampling frequency', 
                        default=config.DEFAULT_SAMPLE_RATE, type=float)


    # preprocess arguments 
    parser.add_argument('--filt-fl', help="Low frequency for bandpass filter", 
                        default=config.DEFAULT_FLOW, nargs='+', type=float)
    parser.add_argument('--filt-fh', help="High frequency for bandpass filter", 
                        default=config.DEFAULT_FHIGH, nargs='+', type=float)
    parser.add_argument('--filt-order', help='Bandpass filter order', 
                        default=config.DEFAULT_FORDER, type=int)
    
    # timeseries arguments 
    parser.add_argument('--train-kernel', help='Length of each segment in seconds', 
                        default=config.DEFAULT_TRAIN_KERNEL, type=float)
    parser.add_argument('--train-stride', help='Stride between segments in seconds', 
                        default=config.DEFAULT_TRAIN_STRIDE, type=float)
    parser.add_argument('--pad-mode', help='Padding mode', 
                        default=config.DEFAULT_PAD_MODE, type=str)

    # training arguments
    parser.add_argument('--batch-size', help='Batch size',
                        default=config.DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument('--max-epochs', help='Maximum number of epochs to train on',
                        default=config.DEFAULT_MAX_EPOCHS, type=int)
    parser.add_argument('--num-workers', help='Number of worker of DataLoader',
                        default=config.DEFAULT_NUM_WORKERS, type=int)
    parser.add_argument('--lr', help='Learning rate of ADAM optimizer', 
                        default=config.DEFAULT_LR, type=float)
    parser.add_argument('--weight-decay', help='Weight decay of ADAM optimizer',
                        default=config.DEFAULT_WEIGHT_DECAY, type=float)
    
    # loss function arguments 
    parser.add_argument('--fftlength', help='FFT length of loss PSD',
                        default=config.DEFAULT_FFT_LENGTH, type=float)
    parser.add_argument('--overlap', help='Overlapping of loss PSD',
                        default=config.DEFAULT_OVERLAP, type=float)

    parser.add_argument('--psd-weight', help='PSD weight of composite loss',
                        default=config.DEFAULT_PSD_WEIGHT, type=float)
    parser.add_argument('--mse-weight', help='MSE weight of composite',
                        default=config.DEFAULT_MSE_WEIGHT, type=float)
    parser.add_argument('--cross-psd-weight', help='Cross-edge PSD weight of comp. loss',
                        default=config.DEFAULT_CROSS_PSD_WEIGHT, type=float)
    parser.add_argument('--edge-weight', help='edge weight of composite',
                        default=config.DEFAULT_EDGE_WEIGHT, type=float)
    parser.add_argument('--edge-frac', help='fraction of the segment considered as edge',
                        default=config.DEFAULT_EDGE_FRAC, type=float)

    # input/output arguments 
    parser.add_argument('--train-dir', help='Path to training directory', 
                        default='.', type=str)
    parser.add_argument('--filename-training', help='Path to training dataset file (h5)', 
                        default=None, type=str)
    parser.add_argument('--filename-validation', help='Path to val dataset file (h5)', 
                        default=None, type=str)
    parser.add_argument('--load-dataset', help='Load training dataset',
                        default=False, type=dc.io.str2bool)
    parser.add_argument('--save-dataset', help='Save training dataset', 
                        default=False, type=dc.io.str2bool)
    parser.add_argument('--initial-checkpoint', help='pretrained model to initialize with', 
                        default=None, type=str)
    parser.add_argument('--log', help='Log file', type=str)
    
    # cuda arguments 
    parser.add_argument('--device', help='Device to use', 
                        default=config.DEFAULT_DEVICE, type=str)
    
    params = parser.parse_args()
    return params 

params = parse_cmd()
pickle.dump({'params': params}, open('dc_transform_train,p', 'wb'))
params = pickle.load(open('dc_transform_train.p', 'rb'))['params']


os.makedirs(params.train_dir, exist_ok=True)
if params.log is not None: 
    params.log = os.path.join(params.train_dir, params.log)
# set up logging to both file and console 
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(params.log, mode='a'),  # Write to file
        logging.StreamHandler()  # Write to console
    ]
)
logging.info('Create training directory: {}'.format(params.train_dir))

BS = 8
device = utils.get_device(params.device)
train_data = ts.TimeSeriesSegmentDataset(params.train_kernel, params.train_stride, pad_mode='median', fs=2048)
val_data = ts.TimeSeriesSegmentDataset(kernel=8, stride=0.25, pad_mode='median', fs=2048)


t0 = 1378403243 

# not using the full 3072s 
train_data.read('/storage/home/hcoda1/3/statachar3/scratch/deepcleanv3/data/combined_data.npz', channels='channels.ini',
    start_time=params.train_t0, end_time=params.train_t0+1536, fs=params.fs)  

val_data.read('/storage/home/hcoda1/3/statachar3/scratch/deepcleanv3/data/combined_data.npz', channels='channels.ini',
    start_time=params.train_t0+1536, end_time=params.train_t0+3072, fs=params.fs) 

# test_data.read('compined_data.npz', channels='channels.ini', 
#     start_time=t0+2560, end_time=t0+3072, fs=2048)

train_data = train_data.bandpass(params.filt_fl, params.filt_fh, params.filt_order, channels='target')
val_data = val_data.bandpass(params.filt_fl, params.filt_fh, params.filt_order, channels='target')
# test_data = test_data.bandpass(110, 130, order=8, channels='target')


# filter pad default from deepclean-prod, is 5: 

train_data.data = train_data.data[:, int(params.filt_pad * params.fs):-int(params.filt_pad * param.fs)]
val_data.data = val_data.data[:, int(params.filt_pad * params.fs):-int(params.filt_pad * params.fs)]
# test_data.data = test_data.data[:, int(filt_pad * fs):-int(filt_pad * fs)]

mean = train_data.mean 
std = train_data.std 
train_data = train_data.normalize()
val_data = val_data.normalize(mean, std)
# test_data = test_data.normalize(mean, std)

# TODO: rebuild windows after .data changes , should restructure this 
train_data.build_windows()
val_data.build_windows()
aux_patch, tgt_patch = train_data[0]
# print(aux_patch.shape, tgt_patch.shape)

# print('train windows: ', len(train_data))
# print('val windows: ', len(val_data))

train_loader = DataLoader(
    train_data,
    batch_size=params.batch_size, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=4, 
    drop_last=True
    )
val_loader = DataLoader(
    val_data, 
    batch_size=params.batch_size, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=4,
    drop_last=True)
    
x, tgt = next(iter(train_loader))
# print('x: ', x.shape)  # (B, C, L) 
# print('tgt: ', tgt.shape) # (B, L)

model = hy.HybridTransformerCNN(C=x.shape[1], fs=params.fs, window_sec=8.0,
                                       d_model=128, nhead=8, num_layers=2,
                                       cnn_kernel=2, cnn_layers=7, n_iters=2)

# model = dc.model.deepclean.DeepClean(train_data.n_channels-1)
model = model.to(device)

# criterion = nn.MSELoss() 
criterion = dc.criterion.CompositePSDLoss(
    fs=params.fs,
    fl=params.filt_fl,
    fh=params.filt_fh,
    fftlength=params.fftlength,
    overlap=None,
    psd_weight=params.psd_weight,
    mse_weight=params.mse_weights,
    reduction='mean',
    device=device,
    average='mean'
)

optimizer = optim.Adam(model.parameters(), lr = params.lr, weight_decay=params.weight_decay)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

train_logger = dc.logger.Logger(outdir=train_dir, metrics=['loss'])
utils.train(
    train_loader, model, criterion, device, optimizer, lr_scheduler, 
    val_loader=val_loader, max_epochs=params.max_epochs, logger=train_logger)


# with torch.no_grad():
    # pred = model(x)
    # print('pred shape: ', pred.shape)
   