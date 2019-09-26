import cv2
import json
import glob
import os.path
import imageio
import numpy as np
import torch

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose

def get_img_path(img_name, nuclei_root):
    name = img_name.split('/')[-1]
    return os.path.join(nuclei_root, img_name, 'image', name + '.png')

def get_instance_label_path(img_name, nuclei_root):
    name = img_name.split('/')[-1]
    return os.path.join(nuclei_root, img_name, 'label', name + '_label.npy')

def get_3cls_label_path(img_name, nuclei_root):
    name = img_name.split('/')[-1]
    return os.path.join(nuclei_root, img_name, '3cls_label', name + '_3cls_label.npy')

def get_watershed_label_path(img_name, nuclei_root):
    name = img_name.split('/')[-1]
    return os.path.join(nuclei_root, img_name, 'watershed_label', name + '_watershed_label.npy')

#def get_pseudo_label_path(img_name, pseudo_label_dir, num_round):
#    name = img_name.split('/')[-1]
#    return os.path.join(pseudo_label_dir, img_name, '.npy')

def load_img_name_json(data_split_list_path, data_type='train'):
    
    img_name_list = []
    with open(data_split_list_path, 'r') as file:
        
        data_list = json.load(file)
        img_name_list = data_list[data_type]
    
    return img_name_list

def to_one_hot(sparse_integers_arr, maximum_val=None, dtype=np.bool):

    if maximum_val is None:
        maximum_val = np.max(sparse_integers_arr) + 1
    
    one_hot = np.eye(maximum_val, dtype=dtype)[sparse_integers_arr]
    
    return one_hot

class NucleiSelfSegDataset(BaseDataset):
    """The nuclei selftraining segmentation dataset.
    Args:
        data_split_list (str): The path of type of the training and validation and test data split list file.
        train_transforms (Box): The preprocessing and augmentation techiques applied to the training data.
        valid_transforms (Box): The preprocessing and augmentation techiques applied to the validation data.
        label_type: The type of the label ('instance', '3cls_label', or 'watershed_label').
        label_proportion: The proportion of the dropped label.
        random_seed: The random seed for the label dropping.
    """
    def __init__(self, data_split_list, train_preprocessings, valid_preprocessings, transforms, augments=None, label_type='3cls_label', label_proportion=0.5, random_seed=0, **kwargs):
        super().__init__(**kwargs)
        self.data_split_list = data_split_list
        self.train_preprocessings = compose(train_preprocessings)
        self.valid_preprocessings = compose(valid_preprocessings)
        self.transforms = compose(transforms)
        self.augments = compose(augments)
        self.label_type = label_type
        self.label_proportion = label_proportion
        self.random_seed = random_seed
        self.img_name_list = load_img_name_json(self.data_split_list, self.type)
    
    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        name = self.img_name_list[index]
        img = imageio.imread(get_img_path(name, self.data_dir))
        img = np.asarray(img)
        ori_img = img.copy()
        if self.label_type=='instance':
            instance_label = np.load(get_instance_label_path(name, self.data_dir))
            
            np.random.seed(self.random_seed)
            label_idxs = np.unique(instance_label)
            np.random.shuffle(label_idxs[1:])
            random_drop_idxs = label_idxs[:int(len(label_idxs)*self.label_proportion)]
            
            semi_label = instance_label.copy()
            semi_label[np.isin(semi_label, random_drop_idxs)] = 0
            semi_label = (semi_label > 0)[...,None].astype(np.int32)
            full_label = (instance_label > 0)[...,None].astype(np.int32)
            
            label_dtype = torch.long
        elif self.label_type=='3cls_label':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            instance_label = np.load(get_instance_label_path(name, self.data_dir))
            tricls_label = np.load(get_3cls_label_path(name, self.data_dir))
            np.random.seed(self.random_seed)
            label_idxs = np.unique(instance_label)
            np.random.shuffle(label_idxs[1:])
            random_drop_idxs = label_idxs[:int(len(label_idxs)*self.label_proportion)]
            random_remain_idxs = label_idxs[int(len(label_idxs)*0.5):]
            
            mask = instance_label.copy()
            mask_dropped = instance_label.copy()
            semi_label = tricls_label.copy()
            
            mask[np.isin(mask, random_drop_idxs)] = 0
            mask_dropped[np.isin(mask_dropped, random_remain_idxs)] = 0
            
            mask = (mask > 0).astype(np.uint8)
            mask_dropped = (mask_dropped > 0 ).astype(np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            #semi_label = to_one_hot(semi_label * mask).astype(np.int32)
            #full_label = to_one_hot(tricls_label).astype(np.int32)
            semi_label = semi_label * mask
            semi_label[np.where((semi_label + mask_dropped)==3)] = 0
            semi_label = semi_label[...,None].astype(np.int32)
            full_label = tricls_label[...,None].astype(np.int32)

            label_dtype = torch.long
        elif self.label_type=='watershed_label':
            instance_label = np.load(get_instance_label_path(name, self.data_dir))
            watershed_label = np.load(get_watershed_label_path(name, self.data_dir))
            
            np.random.seed(self.random_seed)
            label_idxs = np.unique(instance_label)
            np.random.shuffle(label_idxs[1:])
            random_drop_idxs = label_idxs[:int(len(label_idxs)*self.label_proportion)]
            
            mask = instance_label.copy()
            semi_label = watershed_label.copy()
            
            mask[np.isin(mask, random_drop_idxs)] = 0
            mask = (mask > 0).astype(np.uint8)

            semi_label = (semi_label * mask[...,None]).astype(np.float32)
            full_label = watershed_label.astype(np.float32)

            label_dtype = torch.float
        
        if self.type == 'train':            
            ori_img, img ,semi_label, full_label = self.train_preprocessings(ori_img, img, semi_label, full_label, normalize_tags=[False, True, False, False])
            ori_img, img ,semi_label, full_label = self.augments(ori_img, img, semi_label, full_label, interpolation_orders = [1, 1, 0, 0], label_type = self.label_type)
        elif self.type == 'val':
            ori_img, img ,semi_label, full_label = self.valid_preprocessings(ori_img, img, semi_label, full_label, normalize_tags=[False, True, False, False])

        ori_img, img ,semi_label, full_label = self.transforms(ori_img, img ,semi_label, full_label, dtypes=[torch.float, torch.float, label_dtype, label_dtype])
        ori_img, img, semi_label, full_label = ori_img.permute(2, 0, 1).contiguous(), img.permute(2, 0, 1).contiguous(), semi_label.permute(2, 0, 1).contiguous(), full_label.permute(2, 0, 1).contiguous()
        
        return {'name': name, 'ori_image': ori_img, 'image':img, 'semi_label': semi_label, 'full_label': full_label}