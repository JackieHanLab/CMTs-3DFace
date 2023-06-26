import os
import random
import torch
import time
import pandas as pd
import numpy as np
import torch.distributed as dist
from batch_test_train import val
# from pynvml import *
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
from shutil import copyfile
from collections import OrderedDict

#from hydra.experimental import compose, initialize
#from torch.cuda.amp import autocast as autocast

# a trick to get an available port

'''Use hydra to edit global config'''
@hydra.main(config_path='config', config_name='batch_test_config')
def main(cfg:DictConfig) -> None:
    # Main is used to set up GPU configuration for distributed data parallelism (DDP), 
    # it triggers Main_Worker to set up model details
    OmegaConf.set_struct(cfg, False)

    # dataset path, csv format
    cfg.data_path = hydra.utils.to_absolute_path(cfg.dataset)
    if cfg.info_depth.startswith('r'):
        cfg.data_path+='_rela-v'
    #cfg.model_path = hydra.utils.to_absolute_path(cfg.model_path)+'.tar'
    cfg.save_path = hydra.utils.to_absolute_path(cfg.foldername)#+'/'+cfg.model_base+'/'+cfg.load_model)
    cfg.batch_size = int(cfg.basic_batch_size*cfg.loading_multiplier)

    random_seed = random.randint(1, 10000)  # fix random seed (except for numpy) for each run
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    cfg.rank = 0
    cfg.num_classes = 1
    dim_dict = dict(zip(['v','n','c','vn','vc','nc','rvc','rvn','vnc','rvnc','rvh','rvl','rvs','vnh','vnl','vns','vnhl','vnhs','vnls','vnr','vng','vnb','vnrg','vnrb','vngb','rvM','rvO'],\
        [3,3,3,6,6,6,6,6,9,9,4,4,4,7,7,7,8,8,8,7,7,7,8,8,8,4,6]))
    cfg.input_dim = dim_dict[cfg.info_depth]
    #print(OmegaConf.to_yaml(cfg))
    torch.cuda.set_device(cfg.rank)
    #torch.set_default_dtype(torch.float32)
    RegModel = getattr(importlib.import_module('models.{}.model'.format(cfg.model.name)), 'PointTransformer')(cfg)
    
    RegModel = torch.nn.DataParallel(RegModel, device_ids=[cfg.rank])
    RegModel = RegModel.cuda(cfg.rank)
    RegModel = RegModel.eval()
    with torch.no_grad():
        # contents = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'models.txt')).readlines()
        # models = [model.replace('\n','') for model in contents if model.endswith('.tar\n')*(not model.split('/')[-1].startswith('CosAnneal_best_model'))]
        model_home = os.path.join(os.path.dirname(os.path.realpath(__file__)),f'CrossGroup-{cfg.cross_group}')
        models = [os.path.join(model_home, model_name) for model_name in os.listdir(model_home) if ('best' not in model_name)*(model_name.endswith('.tar'))]
        # each_share = len(models)//cfg.Poolsize
        # if cfg.batch_order!=cfg.Poolsize:
        #     models = models[(cfg.batch_order-1)*each_share:cfg.batch_order*each_share]
        # else:
        #     models = models[(cfg.batch_order-1)*each_share:]
        for model in models:
            # try:
            test(RegModel, cfg, model, epoch = 0)
            # except:
            #     print(f'{model} not yet included')


def test(network, args, model_path, epoch):
    # load existing model
    info = model_path.split('/')[-1].split('_')
    #dataset_rank, model_epoch = info[4], info[6][:-4]
    model_epoch = info[-1].split('.')[0].split('-')[-1]
    info_depth = info[0].split('-')[-1]
    print(f'loading model saved at epoch {model_epoch}')
    map_location = f'cuda:{args.rank}'#{'cuda:%d' % 0: 'cuda:%d' % args.rank}
    checkpoint = torch.load(model_path, map_location=map_location)
    #optimized_epoch = checkpoint['epoch']
    trained_state_dict = checkpoint['model_state_dict']
    network.load_state_dict(trained_state_dict)
    #for cat in [f'val_{dataset_rank}','test']:
    # suffix = args.cat
    target = os.path.join(args.save_path, f'InfoDepth-{info_depth}_gpu-2_epoch_{model_epoch}.csv')    
    if os.path.isfile(target):
        print(f'Inference done for {target}')
        return
    else:
        start = time.time()
        val_MAE, ID, Chron, Pred = val(network, args, epoch)
        print(f'ColorSpace-RGB_InfoDepth-{info_depth}_gpu-2_InferAug-{args.aug} at epoch {model_epoch} has MAE {val_MAE}. Evaluation took {time.time()-start} seconds')
        os.makedirs(args.save_path, exist_ok=True)
        pd.DataFrame(np.hstack((ID[...,None], Chron[...,None], Pred[...,None])), columns=['id','chron','pred']).to_csv(target, index=False)
    


if __name__ == '__main__':
    main()


