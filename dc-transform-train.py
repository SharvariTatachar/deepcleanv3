import os
import sys
import torch 
import time
import pickle
import argparse
import json 
import configparser
import logging
import random 
import numpy as np 
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

def load_channels(path):
    with open(path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

def set_seed(seed: int): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id): 
    worker_seed = torch.initial_seed() % 2**32 
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
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
                        default='configs/118dcv3train.ini')
    
    # dataset arguments
    parser.add_argument('--train-t0', help='GPS of the first sample',
                        type=int, default=None)
    parser.add_argument('--train-duration', help='Duration of train/val frame',
                        type=int, default=None)
    # parser.add_argument('--chanslist', help='Path to channel list',
    #                     type=str, default=None)
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
    parser.add_argument('--seed', type=int, default=0)
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
    # if params.chanslist is None and 'chanslist' in c:
    #     params.chanslist = c.get('chanslist')
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

timestamp = int(time.time())
params = parse_cmd()
set_seed(params.seed)  
print(f"Using seed: {params.seed}")
pickle.dump({'params': params}, open('dc_transform_train.p', 'wb'))
params = pickle.load(open('dc_transform_train.p', 'rb'))['params']


# CHANNEL VOCABULARY 
target_channel = "H1:GDS-CALIB_STRAIN"
baseline_channels = [
    ch for ch in load_channels("ogchannels.ini")
    if ch != target_channel
]

all_channels = [
    ch for ch in load_channels("channels.ini")
    if ch != target_channel
]
noisy_pool = [] 
for ch in all_channels: 
    if ch not in baseline_channels: 
        noisy_pool.append(ch)

channel_to_id = {
    ch: i for i, ch in enumerate(all_channels)
}
id_to_channel = { 
    i: ch for ch, i in channel_to_id.items()
}

channel_name_to_data_index = {
    ch: i for i, ch in enumerate(all_channels)
}

print('len all_chans: ', len(all_channels))
print('len baseline: ', len(baseline_channels))
print('len noisy: ', len(noisy_pool))
print('max data index: ', max(channel_name_to_data_index.values()))

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

train_data.read('/storage/home/hcoda1/3/statachar3/r-pli77-0/deepcleanv3/data/combined_data_updated.npz', channels='channels.ini',
    start_time=params.train_t0, end_time=params.train_t0+1536, fs=params.fs)  

val_data.read('/storage/home/hcoda1/3/statachar3/r-pli77-0/deepcleanv3/data/combined_data_updated.npz', channels='channels.ini',
    start_time=params.train_t0+1536, end_time=params.train_t0+3072, fs=params.fs) 


train_data = train_data.bandpass(params.filt_fl, params.filt_fh, params.filt_order, channels='target')
val_data = val_data.bandpass(params.filt_fl, params.filt_fh, params.filt_order, channels='target')


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
g = torch.Generator() 
g.manual_seed(params.seed)

train_loader = DataLoader(
    train_data,
    batch_size=params.batch_size, 
    shuffle=False, 
    generator=g,
    num_workers=4,
    worker_init_fn=seed_worker,
    pin_memory=False, 
    persistent_workers=True, 
    prefetch_factor=4, 
    drop_last=True
    )
val_loader = DataLoader(
    val_data, 
    batch_size=params.batch_size, 
    shuffle=False, 
    generator=g,
    num_workers=4, 
    worker_init_fn=seed_worker,
    pin_memory=False, 
    persistent_workers=True, 
    prefetch_factor=4,
    drop_last=True)
    
x, tgt = next(iter(train_loader))

n_noisy = 10 # TODO: make this a parameter later 
model_C = len(baseline_channels) + n_noisy

selected_channels = baseline_channels + random.sample(noisy_pool, n_noisy)

selected_indices = [
    channel_name_to_data_index[ch]
    for ch in selected_channels
]

selected_ids = torch.tensor(
    [channel_to_id[ch] for ch in selected_channels],
    dtype=torch.long,
    device=device
)

x_sub = x[:, selected_indices, :].to(device)
channel_ids = selected_ids.unsqueeze(0).expand(x_sub.shape[0], -1)

# print("x_sub shape:", x_sub.shape)
# print("channel_ids shape:", channel_ids.shape)
# print("first 5 selected channels:", selected_channels[:5])
# print("first 5 channel ids:", channel_ids[0, :5])
# print('len channel_to_id: ', len(channel_to_id), flush=True) 
# print("max channel id:", max(channel_to_id.values()), flush=True)

# print("Using utils from:", utils.__file__)

model = hy.HybridTransformerCNN(C=model_C, fs=params.fs, window_sec=8.0,
                                       d_model=128, nhead=8, num_layers=3,
                                       cnn_kernel=2, cnn_layers=7, n_iters=2,
                                       num_channel_ids=max(channel_to_id.values())+1,
                                )

# model = dc.model.deepclean.DeepClean(train_data.n_channels-1)
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

# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

train_logger = dc.logger.Logger(outdir=params.train_dir, metrics=['loss'])
history = utils.train(
    train_loader, model, criterion, device, optimizer, lr_scheduler, 
    val_loader=val_loader, max_epochs=params.max_epochs, logger=train_logger, 
    dynamic_channels=True, all_channels=all_channels, 
    baseline_channels=baseline_channels, noisy_pool=noisy_pool, 
    channel_to_id=channel_to_id)

model.eval()

with torch.no_grad():
    x_val, tgt_val = next(iter(val_loader))
    x_val = x_val.to(device)

    selected_idx, selected_names, sampled_noisy = select_chans(
        all_channels, baseline_channels, noisy_pool
    )

    selected_idx_tensor = torch.as_tensor(selected_idx, dtype=torch.long, device=device)

    selected_ids = torch.tensor(
        [channel_to_id[ch] for ch in selected_names],
        dtype=torch.long,
        device=device,
    )

    x_val = x_val[:, selected_idx_tensor, :]
    channel_ids = selected_ids.unsqueeze(0).expand(x_val.size(0), -1)

    pred = model(x_val, channel_ids)    
    attn = model.transformer.get_attention()[-1]
    B = model.last_B
    C = model.last_C
    Tds = model.last_Tds

    attn = attn.view(B, Tds, attn.shape[1], C, C)

#     pred1 = model(x_sub, channel_ids)
#     pred2 = model(x_sub, channel_ids)
#     perm = torch.randperm(x_sub.shape[1], device=device)
#     x_perm = x_sub[:, perm, :]
#     ids_perm = channel_ids[:, perm]
#     pred3 = model(x_perm, ids_perm)
#     diff = (pred1 - pred2).abs().mean().item()
#     diff_perm = (pred1 - pred3).abs().mean().item() 
# print("Repeat difference - no permutation: ", diff)
# print('Permutation consistency diff: ', diff_perm)
    
# run_data = {
#     'model_name': model.__class__.__name__,
#     'batch_size': params.batch_size, 
#     'lr': params.lr, 
#     'weight_decay': params.weight_decay, 
#     'max_epochs': params.max_epochs, 
#     'train_t0': params.train_t0, 
#     'train_duration': params.train_duration,
#     'fs': params.fs, 
#     'filt_fl': params.filt_fl,
#     'filt_fh': params.filt_fh,
#     'history': history
# }

# run_path = os.path.join(params.train_dir, f'permutation')
# with open(run_path, 'w') as f: 
#     json.dump(run_data, f, indent=2)



# with torch.no_grad():
    # pred = model(x)
    # print('pred shape: ', pred.shape)
   