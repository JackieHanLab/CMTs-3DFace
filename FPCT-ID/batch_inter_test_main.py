import os
import random
import torch
import time
import pandas as pd
import numpy as np
from batch_inter_test_train import val
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
from shutil import copyfile

#from hydra.experimental import compose, initialize
#from torch.cuda.amp import autocast as autocast

'''Use hydra to edit global config'''
@hydra.main(config_path='config', config_name='batch_inter_test_config')
def main(cfg:DictConfig) -> None:
    # Main is used to set up GPU configuration for distributed data parallelism (DDP), 
    # it triggers Main_Worker to set up model details
    OmegaConf.set_struct(cfg, False)

    # dataset path, csv format
    cfg.data_path = hydra.utils.to_absolute_path(cfg.dataset)
    cfg.save_path = hydra.utils.to_absolute_path(cfg.foldername)#+'/'+cfg.model_base+'/'+cfg.load_model)
    cfg.batch_size = int(cfg.basic_batch_size*cfg.loading_multiplier)

    random_seed = random.randint(1, 10000)  # fix random seed (except for numpy) for each run
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    cfg.rank = 0
    cfg.num_classes = len(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'DataSplit', f'YourTestDateRecord.txt'), 'r').readlines())
    dim_dict = dict(zip(['v','n','c','vn','vc','nc','rv','rvc','rvn','vnc','rvnc','rvh','rvl','rvs','vnh','vnl','vns','vnhl','vnhs','vnls','vnv','vnhv','vnsv','rvM','rvO','nls'],\
        [3,3,3,6,6,6,3,6,6,9,9,4,4,4,7,7,7,8,8,8,7,8,8,4,6,5]))
    cfg.input_dim = dim_dict[cfg.info_depth]
    #print(OmegaConf.to_yaml(cfg))
    torch.cuda.set_device(cfg.rank)
    #torch.set_default_dtype(torch.float32)
    RegModel = getattr(importlib.import_module('models.{}.model'.format(cfg.model.name)), 'PointTransformer')(cfg)
    RegModel = torch.nn.DataParallel(RegModel, device_ids=[cfg.rank])
    RegModel = RegModel.cuda(cfg.rank)
    RegModel = RegModel.eval()
    with torch.no_grad():
        model_roots = [model_type.path for model_type in os.scandir(cfg.model_home)]
        for model_root in model_roots:
            for model in os.scandir(model_root):
                if model.name.endswith('.tar'):
                    test(RegModel, cfg, model.path, epoch = 0)


def test(network, args, model_path, epoch):
    # load existing model
    info = model_path.split('/')[-1].split('_')
    model_epoch = info[-1].split('.')[0].split('-')[-1]
    print(f'loading model saved at epoch {model_epoch}')
    map_location = f'cuda:{args.rank}'
    checkpoint = torch.load(model_path, map_location=map_location)
    trained_state_dict = checkpoint['model_state_dict']
    network.load_state_dict(trained_state_dict)
    target = os.path.join(args.save_path, f'YourResult.csv')    
    if os.path.isfile(target):
        print(f'Inference done for {target}')
        return
    else:
        start = time.time()
        accuracy, ID, Label, Pred, Embed = val(network, args.cat, args, epoch)
        os.makedirs(args.save_path, exist_ok=True)
        pred_df = pd.DataFrame(np.hstack((ID[...,None], Label[...,None], Pred[...,None])), columns=['id','label','pred'])
        embed_df = pd.DataFrame(Embed, columns=[str(i) for i in range(Embed.shape[1])])
        infer = pd.concat((pred_df,embed_df),axis=1)
        infer.to_csv(target, index=False)
    


if __name__ == '__main__':
    main()


