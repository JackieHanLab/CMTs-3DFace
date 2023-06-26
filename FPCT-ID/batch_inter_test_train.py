import torch
from torch.cuda.amp import autocast as autocast
from batch_inter_test_dataset import ModelNetDataLoader
import torch


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
        CRT=opt.CRT,
        meta=opt.meta, 
        nprseed = epochseed)

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

def val(model, cat, opt, epoch):
    val_loader=load_data(cat, opt, epoch)

    # record val MAE and loss
    val_accuracy = 0.0
    val_total = 0

    # detail cache
    id_cache = []
    label_cache = []
    pred_cache = []
    embed_cache = []
    # data_cache = []

    # evaluate on val set
    
    for data in val_loader:
        id, points, label = data
        points, label = points.cuda(opt.rank), label.cuda(opt.rank)
        pred, embed = model(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(label.data).cpu().sum().item()
        val_accuracy+=correct
        val_total += points.shape[0]
        id_cache.append(id)
        label_cache.append(label)
        pred_cache.append(pred_choice)
        embed_cache.append(embed)
    val_accuracy = val_accuracy/val_total
    all_id = torch.cat(id_cache).to(dtype=torch.int)
    all_label = torch.cat(label_cache)
    all_pred = torch.cat(pred_cache)
    all_embed = torch.cat(embed_cache, dim=0)
    return val_accuracy, all_id.cpu().detach().numpy(), all_label.cpu().detach().numpy(), all_pred.cpu().detach().numpy(), all_embed.cpu().detach().numpy()






