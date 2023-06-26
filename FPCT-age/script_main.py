import os
import random
import torch
import time
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
import socket
from torch.cuda.amp import GradScaler as Scaler
from script_train import train, val
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
from shutil import copyfile
import torch.nn as nn

# a trick to get an available port for distributed data parallelism (DDP)
def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 0))
    sockname = sock.getsockname()
    sock.close()
    return str(sockname[1])

# record data for the convenience of analysis
def record_csv(record_path, epoch, loss, train_MAE, val_MAE, MAE_winner, learning_rate, run_time):
    if os.path.exists(record_path):
        train_n_val_csv = pd.read_csv(record_path)
    else:
        train_n_val_csv = pd.DataFrame(columns=['epoch', 'loss', 'train_MAE', 'val_MAE', 'MAE winner', 'lr', 'run_time'])
    train_n_val_csv.loc[str(epoch + 1)] = [int(epoch+1), loss, train_MAE, val_MAE, MAE_winner, learning_rate, run_time]
    train_n_val_csv.to_csv(record_path, index=False)

'''Use hydra to edit global config'''
@hydra.main(config_path='config', config_name='config')
def main(cfg:DictConfig) -> None:
    # Main is used to set up GPU configuration for DDP, 
    # it calls Main_Worker to set up model details
    OmegaConf.set_struct(cfg, False)

    # dataset path
    cfg.data_path = hydra.utils.to_absolute_path(cfg.dataset)
    cfg.model_path = hydra.utils.to_absolute_path(cfg.model_path)
    cfg.save_path = hydra.utils.to_absolute_path(cfg.foldername)
    cfg.batch_size = int(cfg.basic_batch_size*cfg.loading_multiplier)
    cfg.learning_rate = cfg.basic_learning_rate*cfg.loading_multiplier*cfg.ga_multiplier

    # Record the gpu node used in case systematic difference exists
    try:
        cfg.nodename = os.environ['SLURM_NODELIST']
    except:
        cfg.nodename = 'debuggpu'
    
    # Detect the number of GPU and decide whether to do DDP
    if cfg.ddp:
        cfg.num_gpus = torch.cuda.device_count()
    else:
        cfg.num_gpus = 1
    print('We are using {0} gpus on node {1}'.format(cfg.num_gpus, cfg.nodename))
    if cfg.num_gpus == 1:
        cfg.distributed = 0
        main_worker(0,cfg)
    else:
        cfg.distributed = 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        free_port = find_free_port()           
        os.environ['MASTER_PORT'] = free_port
        print('Using port {0}'.format(free_port))
        mp.spawn(main_worker, nprocs=cfg.num_gpus, args=(cfg,),join=True)


def main_worker(gpu, args):
    # get rank for the current process  
    args.rank = gpu
    print('Spawned process of rank {0}'.format(args.rank))
    
    # fix and output random seed (except for numpy) for each process 
    random_seed = random.randint(1, 10000)
    print('Process rank {0} run with random Seed: {1}'.format(args.rank, random_seed))
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # complete output filename (except for file type) and create folder for log and csv data
    os.makedirs(args.save_path, exist_ok=True)
    if args.info_depth.endswith('c'):
        info_depth = args.info_depth.replace('c',args.color_space.lower())
    else:
        info_depth = args.info_depth
    args.record_name = args.save_path+f'/trainingprocess_InfoDepth-{info_depth}_gpu-{args.num_gpus}_aug-{args.aug}.csv'

    # tell the model to do regress and use specified data fields
    args.num_classes = 1
    dim_dict = dict(zip(['v','n','c','vn','vc','nc','rv','rvc','rvn','vnc','rvnc','rvh','rvl','rvs','vnh','vnl','vns','vnhl','vnhs','vnls','vnv','vnhv','vnsv','rvM','rvO','nls'],\
        [3,3,3,6,6,6,3,6,6,9,9,4,4,4,7,7,7,8,8,8,7,8,8,4,6,5]))
    args.input_dim = dim_dict[args.info_depth]
    
    # record all configurations
    if args.rank==0:
        print(OmegaConf.to_yaml(args))
        with open(f'{args.save_path}/config.txt','w') as f:
            f.write(OmegaConf.to_yaml(args))

    # initialize model
    torch.cuda.set_device(args.rank)
    RegModel = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformer')(args)

    if args.distributed == 1:
        dist.init_process_group(                                   
            backend='nccl',
            init_method="env://",                                                              
            world_size=args.num_gpus,                              
            rank=args.rank)    
        RegModel = nn.SyncBatchNorm.convert_sync_batchnorm(RegModel).cuda(args.rank)
        RegModel = nn.parallel.DistributedDataParallel(RegModel, device_ids=[args.rank])
    else:
        RegModel = nn.DataParallel(RegModel, device_ids=[args.rank])
        RegModel = RegModel.cuda(args.rank)
    
    # initialize optimizer and auto mixed precision (amp)      
    criterion = nn.MSELoss()
    if args.ozername == 'Adam':
        optimizer = torch.optim.Adam(RegModel.parameters(),lr=args.learning_rate,betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.ozername == 'SGD':
        optimizer = torch.optim.SGD(RegModel.parameters(),lr=args.learning_rate, momentum=args.beta1, weight_decay=args.weight_decay)
    scaler = Scaler() if args.amp else None

    # load existing model
    if args.load_model:
        try:
            if args.rank==0:
                print(f'loading {args.model_path}')
            map_location = f'cuda:{args.rank}'
            checkpoint = torch.load(args.model_path, map_location=map_location)
            start_epoch = checkpoint['epoch']
            RegModel.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.rank==0:
                print(f'Using pretrained model, start at epoch {start_epoch}')
        except:
            if args.rank==0:
                print('Fail to load existing model, starting training from scratch...')
            start_epoch = 0
    else:
        print('Not using existing model, starting training from scratch...')
        start_epoch=0

    # check whether parameters are registered and loaded to GPU properly
    if args.rank==0:
        print('Checking GPUnity and trainability...')
        print('Total number of parameters: {0}'.format(sum(param.numel() for param in RegModel.parameters())))
        print('Number of trainable parameters: {0}'.format(sum(param.numel() for param in RegModel.parameters() if param.requires_grad)))
    
        for param in RegModel.parameters():
            if param.is_cuda == False:
                print('{0} is not on GPU'.format(param))
            if param.requires_grad == False:
                print('{0} is not trainable'.format(param))
        print('Parameter property check done')          


    
    # initialize some global variables
    MAE_Winner = 100.0
    num_batches = len(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'DataSplit', f'YourTrainingDataRecord.txt'), 'r').readlines())//args.batch_size
    if 'CosAnneal' in args.scheduler:
        period_factor = int(args.scheduler.split('x')[-1])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batches*period_factor, eta_min=0, last_epoch = -1)
    elif 'OneCycle' in args.scheduler:
        amplitude_factor = int(args.scheduler.split('x')[-1])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=amplitude_factor*args.learning_rate, steps_per_epoch=num_batches, epochs=args.num_epochs, final_div_factor=1e10)
    elif 'Plateau' in args.scheduler:
        patience_factor = int(args.scheduler.split('x')[-1])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=patience_factor, verbose=True, threshold=0.001, eps=1e-16)
    elif 'Step' in args.scheduler:
        step_factor = int(args.scheduler.split('x')[-1])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_factor, gamma=0.3)
    else:
        scheduler = None


    if args.rank==0:
        print('All initiations done')
        # start training
        print('Start training ...')

    start = time.time()
    
    for epoch in range(args.num_epochs):
        
        ave_loss, train_MAE = train(RegModel, criterion, optimizer, scheduler, scaler, args, epoch)
        # record the best MAE in val set
        val_MAE = val(RegModel, args, epoch)
        if 'Step' in args.scheduler:
            scheduler.step()
        elif 'Plateau' in args.scheduler:
            scheduler.step(train_MAE)
        if args.rank == 0:
            if val_MAE<MAE_Winner:
                MAE_Winner = val_MAE
            record_csv(args.record_name, start_epoch+epoch, ave_loss, train_MAE, val_MAE, MAE_Winner, optimizer.param_groups[0]['lr'], time.time()-start)
            # start to save model checkpoint after 200 epochs
            if args.save*(start_epoch+epoch>=200):
                state_dict = RegModel.state_dict()
                latest_model_path = f'{args.save_path}/InfoDepth-{info_depth}_gpu-{args.num_gpus}_aug-{args.aug}_epoch-{epoch+1}.tar'
                torch.save({
                        'epoch': start_epoch+epoch+1,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': ave_loss
                    }, latest_model_path)
                try:
                    os.remove(os.path.join(args.save_path, f'best_model_InfoDepth-{info_depth}_gpu-{args.num_gpus}_aug-{args.aug}.tar'))
                except:
                    pass
                copyfile(latest_model_path, os.path.join(args.save_path, f'best_model_InfoDepth-{info_depth}_gpu-{args.num_gpus}_aug-{args.aug}.tar'))
            print(f'{epoch+1}/{args.num_epochs} epochs finished on gpu {args.rank}, InfoDepth-{info_depth}_gpu-{args.num_gpus}_aug-{args.aug} repeat {args.repeat}| \
                training loss: {ave_loss}; train set MAE: {train_MAE}; val set MAE: {val_MAE}; MAE winner: {MAE_Winner}; current learning rate: {optimizer.param_groups[0]["lr"]}; \
                    running time: {time.time()-start}')
            
    if args.rank==0:
        print('End of training...')
        

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    main()


