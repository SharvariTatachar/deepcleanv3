import os
import logging

logger = logging.getLogger(__name__)

import numpy as np

import torch

from ..logger import Logger

def train(train_loader, model, criterion, device, optimizer, lr_scheduler, 
        val_loader=None, max_epochs=10, logger=None): 
        
        # If Logger is not given, create a default logger
        if logger is None:
            logger = Logger(outdir='train_dir', metrics=['loss'])
        
        # Start training
        num_batches = len(train_loader)
        model.train()
        
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda"))
        for epoch in range(max_epochs): 
            # Training phase
            model.train()
            train_loss_sum = torch.zeros((), device=device)
            n_seen = 0 
            for step, (x, tgt) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)  # (B, C, L)
                tgt = tgt.to(device, non_blocking=True)  # (B, L)

                optimizer.zero_grad(set_to_none=True) 
                with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
                    pred = model(x).squeeze(1) # (B, L)
                    loss = criterion(pred, tgt)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Accumulate training loss
                if hasattr(criterion, 'reduction') and criterion.reduction == 'mean':
                    train_loss_sum += loss.detach() * x.size(0)
                    n_seen += x.size(0)
                else:
                    train_loss_sum += loss.detach()
                    n_seen +=1 
                    
                # if step % 10 == 0: 
                #     logging.info(f"epoch {epoch} step {step} loss {loss.item():.6f}")


            # Compute average training loss
            train_loss = (train_loss_sum/n_seen).item()
            
            # Validation phase (if val_loader is provided)
            val_loss = 0.0
            if val_loader is not None:
                model.eval()
                val_loss_sum = torch.zeros((), device=device)
                n_val = 0
                with torch.no_grad():
                    for x_val, tgt_val in val_loader:
                        x_val = x_val.to(device, non_blocking=True)
                        tgt_val = tgt_val.to(device, non_blocking=True)
                        with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
                            pred_val = model(x_val).squeeze(1)
                            loss_val = criterion(pred_val, tgt_val)
                        if hasattr(criterion, 'reduction') and criterion.reduction == 'mean':
                            val_loss_sum += loss_val.detach() * x_val.size(0)
                            n_val += x_val.size(0)
                        else:
                            val_loss_sum += loss_val.detach()
                            n_val += 1
                val_loss = (val_loss_sum/n_val).item()
                model.train()
            
            # Update LR scheduler at end of epoch
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # Update metrics with logger
            last_batch_idx = num_batches - 1
            logger.update_metric(train_loss, val_loss, 'loss', epoch, 
                                 last_batch_idx, num_batches)
            
            # Display status
            logger.display_status(epoch, max_epochs, last_batch_idx, num_batches,
                                  train_loss, val_loss, 'loss')
            
            # Log and plot metrics (creates/updates the plot file)
            logger.log_metric()
            
            # Save model checkpoint
            logger.save_model(model, epoch)
            
        logging.info(f"Training completed. Final train loss: {train_loss:.6f}")

def get_device(device):
    ''' Convenient function to set up hardware '''
    if device.lower() == 'cpu':
        device = torch.device('cpu')
    elif device.lower() == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info(f'-Use device: {device}')
        else:
            logging.warning('No MPS available. Use CPU instead.')
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
        logger.info("device count: {}".format(torch.cuda.device_count()))
        logger.info('- Total memory: {:.4f} GB'.format(total_memory))
    else:
        logger.info('- Use device: CPU')
    return device

