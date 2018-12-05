import torch
from torchvision import datasets, transforms


import sys
import time
import random
# import theano
import numpy as np
import traceback
from PIL import Image
from six.moves import queue
from multiprocessing import Process, Event

from lib.imgVoxDataloader import imgVoxDataloader

# from lib.config import cfg
# from lib.data_augmentation import preprocess_img
# from lib.data_io import get_voxel_file, get_rendering_file
from lib.binvox_rw import read_as_3d_array
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')

data_root = './data/'
# train_root = data_root + 'train'
# val_root = data_root + 'val'
# test_root = data_root + 'test'

rendering_root = data_root + 'Rendering'
voxel_root = data_root + 'voxel'

def get_render_file(category, model_id):
    return rendering_root % (category, model_id)


def get_voxel_file(category, model_id):
    return voxel_root % (category, model_id)

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# vox_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
#     ])

def construct_transformer():
    """construct transformer for images"""
    mean = [0.45486851, 0.43632515, 0.40461355]
    std = [0.26440552, 0.26142306, 0.27963778]
    transformer = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transformer

# load the image transformer
centre_crop = transforms.Compose([
    transforms.RandomResizedCrop(144),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_transform = construct_transformer()

def vox_transform(z):
    return z

vox_transform_f = vox_transform
def npyLoader(f):
    x = np.load(f)
    return torch.from_numpy(x)
#     return np.load(f)

# train_dataset = datasets.ImageFolder(root=train_root, transform=base_transform)
# val_dataset = datasets.ImageFolder(root=val_root, transform=base_transform)
# test_dataset = datasets.ImageFolder(root=test_root, transform=base_transform)

ren_dataset = datasets.ImageFolder(root=rendering_root, transform=base_transform)
vox_dataset = datasets.DatasetFolder(root=voxel_root, transform=None,loader=npyLoader,extensions='npy')



torch.backends.cudnn.deterministic=True

def get_ren_data_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
            ren_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader

def get_vox_data_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
            vox_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader

def get_train_data_loaders(idx):
    
    sub_ren_dataset = torch.utils.data.Subset(ren_dataset,idx)
    sub_vox_dataset = torch.utils.data.Subset(vox_dataset,idx)

    render_loader = torch.utils.data.DataLoader(
            sub_ren_dataset, batch_size=len(idx), shuffle=False, num_workers=4, drop_last=True)
    voxel_loader = torch.utils.data.DataLoader(
            sub_vox_dataset, batch_size=len(idx), shuffle=False, num_workers=4, drop_last=True)
    return (render_loader, voxel_loader)

# def get_data_loaders(batch_size):
#     train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = torch.utils.data.DataLoader(
#             val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     return (train_loader, val_loader)

# def get_val_test_loaders(batch_size):
#     val_loader = torch.utils.data.DataLoader(
#             val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     test_loader = torch.utils.data.DataLoader(
#             test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     return (val_loader, test_loader)
