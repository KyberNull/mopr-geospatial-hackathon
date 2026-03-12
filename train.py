"""Training loop for the segmentation model."""

import logging
import os
from losses import dice_loss, compute_means
from model import UNet
import signal
import sys
import torch
from torch import nn, optim, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from transforms import TrainTransforms, EvalTransforms
from utils import get_adamw_param_groups, save_checkpoint, device_setup, setup_logging


###-------CONSTANTS-------###
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
WARMUP_EPOCHS = 5
MODEL_PATH = "model.pt"
NUM_BATCHES = 32
NUM_CLASSES = 21
NUM_EPOCHS = 300
NUM_WORKERS = min(4, os.cpu_count() or 1)
VAL_INTERVAL = 10
NUM_VAL_SAMPLES = 280
###-----------------------###

shutdown_requested = False
pin_memory = False
amp_dtype = torch.bfloat16
logger = logging.getLogger(__name__)

def handle_shutdown(sig, frame):
    global shutdown_requested
    logger.warning(f'Shutdown requested! Signal: {sig}')
    shutdown_requested = True

def validate(model, validation_loader, device, criterion):
    '''Validating the model continuously to prevent overfitting.'''
    model.eval()
    running_val_loss = 0.0
    total_iou = 0.0
    val_iterator = iter(validation_loader)
    with torch.no_grad():
        for i in range(NUM_VAL_SAMPLES):
            try:
                val_input, val_output = next(val_iterator)
            except StopIteration:
                val_iterator = iter(validation_loader)
                val_input, val_output = next(val_iterator)

            val_input, val_output = val_input.to(device, non_blocking=True), val_output.squeeze(1).to(device, non_blocking=True).long()

            with autocast(device_type=device.type, dtype=amp_dtype):
                val_prediction = model(val_input)
                val_loss = criterion(val_prediction, val_output)

            running_val_loss += val_loss.item()

            _, iou = compute_means(val_prediction, val_output, NUM_CLASSES)
            total_iou += iou.item()

        total_iou /= NUM_VAL_SAMPLES

        running_val_loss /= NUM_VAL_SAMPLES
            
        logging.info(f"mCEL: {running_val_loss:.4f}")
        logging.info(f"mIoU: {total_iou:.4f}")

        model.train()
        freeze_encoder(model)

def load_checkpoint(model_path, model, optimizer, scheduler, scaler):
    start_epoch = 0
    try:
        # Resume model, optimizer, scheduler, and scaler states to continue training seamlessly.
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    except FileNotFoundError:
        logger.warning("Checkpoint not found. Training from scratch.")
    except (RuntimeError, KeyError) as err:
        logger.error(f"Checkpoint incompatible with current model architecture: {err}")
        logger.warning("Training from scratch.")
    return model, optimizer, scheduler, scaler, start_epoch
        
def load_dataloader(image_set, transforms, shuffle):
    dataset = datasets.VOCSegmentation('./data', year = '2012', image_set = image_set, transforms = transforms)
    dataLoader = DataLoader(dataset=dataset, batch_size=NUM_BATCHES, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0)
    return dataLoader

def main(device, model_path):
    # GradScaler is only useful on CUDA where float16 gradients can underflow.
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))

    trainLoader = load_dataloader(image_set='train', transforms=TrainTransforms(), shuffle=True)
    validationLoader = load_dataloader(image_set='val', transforms=EvalTransforms(), shuffle=False)

    model = UNet(num_classes=NUM_CLASSES).to(device=device, non_blocking=True)
    freeze_encoder(model)

    optimizer = optim.AdamW(get_adamw_param_groups(model, LEARNING_RATE, LEARNING_RATE/10, WEIGHT_DECAY), lr=LEARNING_RATE)
    model = torch.compile(model)
    warmup_steps = min(max(0, WARMUP_EPOCHS * len(trainLoader)), max(0, NUM_EPOCHS * len(trainLoader) - 1))

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS * len(trainLoader) - warmup_steps,
            eta_min=LEARNING_RATE * 0.1
        )
    
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
    )

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    model, optimizer, scheduler, scaler, start_epoch = load_checkpoint(model_path, model, optimizer, scheduler, scaler)

    model.train()
    freeze_encoder(model)
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_bar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True, disable=not sys.stdout.isatty(), position=0)
        running_loss = 0.0
        for batch, (input, output) in enumerate(epoch_bar):
            if shutdown_requested:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, MODEL_PATH)
                logger.info("Checkpoint saved. Gracefully exiting...")
                return

            # Masks come as [N, 1, H, W]; CrossEntropyLoss expects [N, H, W] class ids.
            input, output = input.to(device, non_blocking=True), output.squeeze(1).to(device, non_blocking=True).long()

            # set_to_none avoids unnecessary memory writes compared to zeroing tensors.
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=amp_dtype):
                prediction = model(input)
                loss = criterion(prediction, output)
                
                # Combine pixel-wise CE with overlap-aware Dice for better segmentation quality.
                loss += dice_loss(prediction, output, NUM_CLASSES)
            
            scaler.scale(loss).backward()
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after_step = scaler.get_scale()

            if scale_after_step >= scale_before_step:
                scheduler.step()

            running_loss += loss.item()

            avg_loss = running_loss / (batch + 1)
            epoch_bar.set_postfix(loss=avg_loss, lr=scheduler.get_last_lr()[0]) # Shows AVG Loss NOT Batch Loss
        
        # Validation loop
        if (epoch + 1) % VAL_INTERVAL == 0:
            validate(model, validationLoader, device, criterion)
        
        #TODO: Maybe save best model and current model
        save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, path=MODEL_PATH)
    logger.info("Training complete. Checkpoint saved.")

if __name__ == "__main__":

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    device, pin_memory, amp_dtype = device_setup()
    setup_logging()

    main(device, MODEL_PATH)