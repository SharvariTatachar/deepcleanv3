import os
import torch 
import pickle
import argparse
import json 
import configparser
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ('yes', 'true', 't', '1')
    return bool(v)


def parse_cmd():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), usage='%(prog)s [options]')

    parser.add_argument('--config', help='Path to .ini config', type=str,
                        default='configs/118train.ini')
    
    # dataset arguments
    parser.add_argument('--train-t0', help='GPS of the first sample',
                        type=int, default=None)
    parser.add_argument('--train-duration', help='Duration of train/val frame',
                        type=int, default=None)
    parser.add_argument('--chanslist', help='Path to channel list',
                        type=str, default=None)
    parser.add_argument('--fs', help='Sampling frequency',
                        type=float, default=None)


    # preprocess arguments
    parser.add_argument('--filt-fl', help="Low frequency for bandpass filter",
                        nargs='+', type=float, default=None)
    parser.add_argument('--filt-fh', help="High frequency for bandpass filter",
                        nargs='+', type=float, default=None)
    parser.add_argument('--filt-order', help='Bandpass filter order',
                        type=int, default=None)
    parser.add_argument('--filt-pad', help='Padding (in seconds) removed after filtering',
                        type=float, default=5.0)
    
    # timeseries arguments
    parser.add_argument('--train-kernel', help='Length of each segment in seconds',
                        type=float, default=None)
    parser.add_argument('--train-stride', help='Stride between segments in seconds',
                        type=float, default=None)
    parser.add_argument('--pad-mode', help='Padding mode',
                        type=str, default=None)

    # training arguments
    parser.add_argument('--batch-size', help='Batch size',
                        type=int, default=None)
    parser.add_argument('--max-epochs', help='Maximum number of epochs to train on',
                        type=int, default=None)
    parser.add_argument('--num-workers', help='Number of workers of DataLoader',
                        type=int, default=None)
    parser.add_argument('--lr', help='Learning rate of ADAM optimizer',
                        type=float, default=None)
    parser.add_argument('--weight-decay', help='Weight decay of ADAM optimizer',
                        type=float, default=None)
    
    # loss function arguments
    parser.add_argument('--fftlength', help='FFT length of loss PSD',
                        type=float, default=None)
    parser.add_argument('--overlap', help='Overlapping of loss PSD',
                        type=float, default=None)

    parser.add_argument('--psd-weight', help='PSD weight of composite loss',
                        type=float, default=None)
    parser.add_argument('--mse-weight', help='MSE weight of composite',
                        type=float, default=None)
    parser.add_argument('--cross-psd-weight', help='Cross-edge PSD weight of comp. loss',
                        type=float, default=0.0)
    parser.add_argument('--edge-weight', help='Edge weight of composite',
                        type=float, default=0.0)
    parser.add_argument('--edge-frac', help='Fraction of the segment considered as edge',
                        type=float, default=0.0)

    # input/output arguments
    parser.add_argument('--train-dir', help='Path to training directory',
                        type=str, default=None)
    parser.add_argument('--filename-training', help='Path to training dataset file (h5)', 
                        default=None, type=str)
    parser.add_argument('--filename-validation', help='Path to val dataset file (h5)',
                        default=None, type=str)
    parser.add_argument('--load-dataset', help='Load training dataset',
                        default=None, type=str2bool)
    parser.add_argument('--save-dataset', help='Save training dataset',
                        default=None, type=str2bool)
    parser.add_argument('--initial-checkpoint', help='pretrained model to initialize with', 
                        default=None, type=str)
    parser.add_argument('--log', help='Log file', type=str, default=None)

    # cuda arguments
    parser.add_argument('--device', help='Device to use',
                        type=str, default=None)
    
    params = parser.parse_args()

    # load .ini
    cfg = configparser.ConfigParser()
    cfg.read(params.config)
    c = cfg['config']

    # override missing CLI args from config file
    if params.train_dir is None and 'train_dir' in c:
        params.train_dir = c.get('train_dir')
    if params.log is None and 'log' in c:
        params.log = c.get('log')
    if params.device is None and 'device' in c:
        params.device = c.get('device')
    if params.train_t0 is None and 'train_t0' in c:
        params.train_t0 = c.getint('train_t0')
    if params.train_duration is None and 'train_duration' in c:
        params.train_duration = c.getint('train_duration')
    if params.fs is None and 'fs' in c:
        params.fs = c.getint('fs')
    if params.chanslist is None and 'chanslist' in c:
        params.chanslist = c.get('chanslist')
    if params.train_kernel is None and 'train_kernel' in c:
        params.train_kernel = c.getfloat('train_kernel')
    if params.train_stride is None and 'train_stride' in c:
        params.train_stride = c.getfloat('train_stride')
    if params.pad_mode is None and 'pad_mode' in c:
        params.pad_mode = c.get('pad_mode')
    if params.filt_fl is None and 'filt_fl' in c:
        # nargs='+' expects a list
        params.filt_fl = [c.getfloat('filt_fl')]
    if params.filt_fh is None and 'filt_fh' in c:
        params.filt_fh = [c.getfloat('filt_fh')]
    if params.filt_order is None and 'filt_order' in c:
        params.filt_order = c.getint('filt_order')
    if params.batch_size is None and 'batch_size' in c:
        params.batch_size = c.getint('batch_size')
    if params.max_epochs is None and 'max_epochs' in c:
        params.max_epochs = c.getint('max_epochs')
    if params.num_workers is None and 'num_workers' in c:
        params.num_workers = c.getint('num_workers')
    if params.lr is None and 'lr' in c:
        params.lr = c.getfloat('lr')
    if params.weight_decay is None and 'weight_decay' in c:
        params.weight_decay = c.getfloat('weight_decay')
    if params.fftlength is None and 'fftlength' in c:
        params.fftlength = c.getfloat('fftlength')
    if params.psd_weight is None and 'psd_weight' in c:
        params.psd_weight = c.getfloat('psd_weight')
    if params.mse_weight is None and 'mse_weight' in c:
        params.mse_weight = c.getfloat('mse_weight')
    if params.save_dataset is None and 'save_dataset' in c:
        params.save_dataset = c.getboolean('save_dataset')
    if params.load_dataset is None and 'load_dataset' in c:
        params.load_dataset = c.getboolean('load_dataset')

    # final fallbacks
    if params.train_dir is None:
        params.train_dir = '.'
    if params.fs is None:
        params.fs = 2048

    return params 

params = parse_cmd()
pickle.dump({'params': params}, open('dc_transform_train.p', 'wb'))
params = pickle.load(open('dc_transform_train.p', 'rb'))['params']


os.makedirs(params.train_dir, exist_ok=True)
if params.log is not None: 
    params.log = os.path.join(params.train_dir, params.log)
# set up logging to both file and console 
if params.log is not None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(params.log, mode='a'),  # Write to file
            logging.StreamHandler()  # Write to console
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler()]
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

train_data.data = train_data.data[:, int(params.filt_pad * params.fs):-int(params.filt_pad * params.fs)]
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

# model = hy.HybridTransformerCNN(C=x.shape[1], fs=params.fs, window_sec=8.0,
#                                        d_model=128, nhead=8, num_layers=2,
#                                        cnn_kernel=2, cnn_layers=7, n_iters=2)

model = dc.model.deepclean.DeepClean(train_data.n_channels-1)
model = model.to(device)

# criterion = nn.MSELoss() 
criterion = dc.criterion.CompositePSDLoss(
    fs=params.fs,
    fl=params.filt_fl,
    fh=params.filt_fh,
    fftlength=params.fftlength,
    overlap=params.overlap,
    psd_weight=params.psd_weight,
    mse_weight=params.mse_weight,
    reduction='mean',
    device=device,
    average='mean'
)

optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

train_logger = dc.logger.Logger(outdir=params.train_dir, metrics=['loss'])
history = utils.train(
    train_loader, model, criterion, device, optimizer, lr_scheduler, 
    val_loader=val_loader, max_epochs=params.max_epochs, logger=train_logger)

run_data = {
    'model_name': model.__class__.__name__,
    'batch_size': params.batch_size, 
    'lr': params.lr, 
    'weight_decay': params.weight_decay, 
    'max_epochs': params.max_epochs, 
    'train_t0': params.train_t0, 
    'train_duration': params.train_duration,
    'fs': params.fs, 
    'filt_fl': params.filt_fl,
    'filt_fh': params.filt_fh,
    'history': history
}

run_path = os.path.join(params.train_dir, 'dc_run1.json')
with open(run_path, 'w') as f: 
    json.dump(run_data, f, indent=2)



# with torch.no_grad():
    # pred = model(x)
    # print('pred shape: ', pred.shape)
   