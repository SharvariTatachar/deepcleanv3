import os
import logging

logger = logging.getLogger(__name__)

import numpy as np

import torch

from ..logger import Logger

def train(train_loader, model, criterion, device, optimizer, lr_scheduler, 
        val_loader=None, max_epochs=2, logger=None): 

        # Two epochs for now 
        for epoch in range(max_epochs): 
            running = 0.0 
            for step, (x, tgt) in enumerate(train_loader):
                x = x.to(device)  # (B, C, T)
                tgt = tgt.to(device)  # (B, T)

                pred = model(x).squeeze(1) # (B, T)
                loss = criterion(pred, tgt)
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step()

                running += loss.item() 
                if step % 10 == 0: 
                    print(f"epoch {epoch} step {step} loss {loss.item():.6f}")

            print((f"epoch {epoch} avg loss {running/(step +1):.6f}"))

def get_device(device):
    ''' Convenient function to set up hardward '''
    if device.lower() == 'cpu':
        device = torch.device('cpu')
    elif device.lower() == 'mps': 
        if torch.backends.mps.is_available(): 
            device = torch.device('mps')
            logger.info(f'-Use device: {device}')
        else: 
            logging.warning('No mps available. Use CPU instead.')
            device = torch.device('cpu')
    elif 'cuda' in device.lower():
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            logging.warning('No GPU available. Use CPU instead.')
            device = torch.device('cpu')
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory *= 1e-9 # convert bytes to Gb
        logger.info('- Use device: {}'.format(torch.cuda.get_device_name(device)))
        logger.info('- Total memory: {:.4f} GB'.format(total_memory))
    else:
        logger.info('- Use device: CPU -- TODO: change this')
    return device
