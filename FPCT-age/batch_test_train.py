import argparse
import random
import time
#from pynvml import *
import os
import torch
import numpy as np
from torch.cuda.amp import autocast as autocast
from batch_test_dataset import ModelNetDataLoader
import torch
import datetime
from pathlib import Path
#from tqdm import tqdm
import sys
import provider
from contextlib import nullcontext

def load_data(cat, opt, epoch):
    # set a different random seed for each epoch so as to load each face in a slightly different way for augmentation
    epochseed = epoch#np.random.randint(0,10000)
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

    #shuffle_dict = {'train':True, 'val':False, 'test':False}
    # define sampler for DDP
    datasampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False)
    
    return dataloader

def val(model, opt, epoch):
    val_loader=load_data('test', opt, epoch)

    # record val MAE and loss
    val_MAE = 0.0
    val_total = 0

    # detail cache
    id_cache = []
    age_cache = []
    pred_cache = []

    # evaluate on val set
    
    for data in val_loader:
        id, points, age = data
        points, age = points.cuda(opt.rank), age.cuda(opt.rank)
        pred = model(points).squeeze(-1)
        val_MAE += torch.sum(torch.abs(pred.data-age.data)).item()
        val_total += points.shape[0]
        id_cache.append(id)
        age_cache.append(age)
        pred_cache.append(pred)
    val_MAE = val_MAE/val_total
    #del val_set, val_sampler, val_loader, points, target, pred, pred_choice
    all_id = torch.cat(id_cache).to(dtype=torch.int)
    all_age = torch.cat(age_cache)
    all_pred = torch.cat(pred_cache)
    return val_MAE, all_id.cpu().detach().numpy(), all_age.cpu().detach().numpy(), all_pred.cpu().detach().numpy()






