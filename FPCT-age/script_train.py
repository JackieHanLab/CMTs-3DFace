import torch
import numpy as np
from torch.cuda.amp import autocast as autocast
from dataset import ModelNetDataLoader
import torch
from contextlib import nullcontext
import torch.distributed as dist

def load_data(cat, opt, epoch):
    # set a different random seed for each epoch so as to load each face in a slightly different way for augmentation
    epochseed = epoch
    dataset = ModelNetDataLoader(root=opt.data_path, 
        epoch = epoch,
        aug = opt.aug,
        color_space = opt.color_space,
        info_depth = opt.info_depth,
        num_points=opt.num_points, 
        category=cat,
        cross_group = opt.cross_group,
        meta=opt.meta, 
        nprseed = epochseed)

    # define sampler for DDP
    if opt.distributed == 1:
        datasampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=opt.num_gpus,
        rank=opt.rank,
        shuffle=True)
        datasampler.set_epoch(epoch)
    else:
        datasampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=(datasampler is None))    
    return dataloader


def train(model, criterion, optimizer, scheduler, scaler, opt, epoch):
    train_loader = load_data('train', opt, epoch)

    # record train set MAE, loss
    train_MAE = 0.0
    train_loss = 0.0
    train_total = 0
    
    # train
    model = model.train()
    for i, data in enumerate(train_loader, 0):
        ga_context = model.no_sync if opt.distributed else nullcontext
        _, points, age = data
        points, age = points.cuda(opt.rank), age.cuda(opt.rank)
        # use gradient accumulation and amp
        if opt.amp:
            with ga_context():
                with autocast():
                    pred = model(points).squeeze(-1)
                    loss = criterion(pred, age)
                    train_loss += loss.item()
                    train_MAE += torch.sum(torch.abs(pred.data-age.data)).item()
                    train_total += points.shape[0]       
                scaler.scale(loss).backward()
            if i % opt.ga_multiplier == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            with ga_context():
                pred = model(points).squeeze(-1)
                loss = criterion(pred, age)
                train_loss += loss.item()
                train_MAE += torch.sum(torch.abs(pred.data-age.data)).item()
                train_total += points.shape[0]
                loss.backward()
            if i % opt.ga_multiplier == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if ('CosAnneal' in opt.scheduler)+('OneCycle' in opt.scheduler):
            scheduler.step()

    ave_loss = train_loss / train_total
    train_MAE = train_MAE / train_total
    
    return ave_loss, train_MAE


def val(model, opt, epoch):
    val_loader=load_data('val', opt, epoch)

    # record val MAE and loss
    val_MAE = 0.0
    val_total = 0

    # evaluate on val set
    model = model.eval()
    with torch.no_grad():
        for data in val_loader:
            _, points, age = data
            points, age = points.cuda(opt.rank), age.cuda(opt.rank)
            pred = model(points).squeeze(-1)
            val_MAE += torch.sum(torch.abs(pred.data-age.data))
            val_total += points.shape[0]
        val_MAE = val_MAE/val_total
        if opt.num_gpus>1:
            dist.reduce(val_MAE,0)
            val_MAE/=opt.num_gpus
        if opt.rank == 0:
            return val_MAE.item()
        else:
            return None





