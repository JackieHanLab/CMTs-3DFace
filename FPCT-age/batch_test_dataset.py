import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, index_points, pc_normalize
import pandas as pd
import argparse

def shuffle_points(item_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(item_data.shape[0])
    np.random.shuffle(idx)
    return item_data[idx]

def rotate_points(item_data, angle_sigmas=np.pi):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    angles = np.random.uniform(-angle_sigma, angle_sigma, 3)
    Rx = np.array([[1,0,0],
                    [0,np.cos(angles[0]),-np.sin(angles[0])],
                    [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                    [0,1,0],
                    [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                    [np.sin(angles[2]),np.cos(angles[2]),0],
                    [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    item_data[:,:3] = np.dot(item_data[:,:3], R)
    item_data[:,3:6] = np.dot(item_data[:,3:6], R)
    return item_data


def shift_points(item_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    item_data[:,:3] += shifts 

    return item_data


def scale_points(item_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    scalev = np.random.uniform(scale_low, scale_high)
    scalen = np.random.uniform(scale_low, scale_high)
    item_data[:,:3] *= scalev
    item_data[:,3:6] *= scalen
    return item_data



class ModelNetDataLoader(Dataset):
    def __init__(self, root, epoch, aug = True, color_space = 'RGB', info_depth='vnc', num_points=12288, category='test', cross_group=1, meta='YourMeta.csv', nprseed=0):
        self.root = root
        self.aug = aug
        self.color_space = color_space
        self.info_depth = info_depth
        self.npoints = num_points
        self.category = category
        self.cross_group = cross_group
        self.nprseed = nprseed
        self.facedirlist = []
        self.meta = meta
        self.data_source = self.root
        suffix = self.category if self.category=='test' else f'{self.category}-{self.cross_group}'
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'DataSplit', f'YourDataRecord.txt'), 'r') as f:
            for line in f:
                if len(line)>0:
                    self.facedirlist.append(os.path.join(self.data_source, line.strip()+'.csv'))
        d = pd.read_csv(os.path.join(root, self.meta))
        self.agedict = dict(zip(d.id,d.age))

    def __len__(self):
        return len(self.facedirlist)

    def __getitem__(self, index):
        np.random.seed(self.nprseed)
        facedir = self.facedirlist[index]
        faceID = os.path.split(facedir)[-1].split('.')[0]
        age = self.agedict[faceID]
        facedata = pd.read_csv(facedir).values
        if facedata.shape[0]>self.npoints:
            choice = np.random.choice(facedata.shape[0], self.npoints, replace=False)
            facedata = facedata[choice, :]
        if self.color_space == 'RGB':
            facedata[:,[-3,-1]] = facedata[:,[-1,-3]]
        
        if self.aug:
            facedata = scale_points(shift_points(rotate_points(shuffle_points(facedata))))
        
        # choose which and how informations are combined
        padding_block = np.array([]).reshape(self.npoints,0)
        first_block = facedata[:,:3] if self.info_depth.startswith('v') else padding_block
        
        if 'n' in self.info_depth:
            second_block = facedata[:,3:6]
        else:
            second_block = padding_block
        
        if self.info_depth.endswith('c'):
            third_block = facedata[:,6:]
        elif self.info_depth.endswith('hl')+self.info_depth.endswith('rg')+self.info_depth.endswith('hs'):
            third_block = facedata[:,-3:-1]
        elif self.info_depth.endswith('hs')+self.info_depth.endswith('rb')+self.info_depth.endswith('hv'):
            third_block = np.hstack((facedata[:,-3:-2], facedata[:,-1:]))
        elif self.info_depth.endswith('ls')+self.info_depth.endswith('gb')+self.info_depth.endswith('sv'):
            third_block = facedata[:,-2:]
        elif self.info_depth.endswith('h')+self.info_depth.endswith('r'):
            third_block = facedata[:,-3:-2]
        elif self.info_depth.endswith('l')+self.info_depth.endswith('g')+self.info_depth.endswith('s'):
            third_block = facedata[:,-2:-1]
        elif self.info_depth.endswith('s')+self.info_depth.endswith('b')+self.info_depth.endswith('v')*(len(self.info_depth)>1):
            third_block = facedata[:,-1:]
        else:
            third_block = padding_block
            
        pts = np.hstack((first_block, second_block, third_block))
        point_set = torch.from_numpy(pts)

        return torch.as_tensor(int(''.join(faceID.split('_')))), point_set.to(dtype=torch.float), torch.as_tensor(age,dtype=torch.float)



