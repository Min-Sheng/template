import csv
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from box import Box
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import random

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose
import src


def _get_instance(module, config, *args):
    """
    Args:
        module (module): The python module.
        config (Box): The config to create the class object.
    Returns:
        instance (object): The class object defined in the module.
    """
    cls = getattr(module, config.name)
    kwargs = config.get('kwargs')
    return cls(*args, **config.kwargs) if kwargs else cls(*args)

def _get_item(module, index):
    image_path, label_path = module.data_paths[index]
    image = nib.load(str(image_path)).get_data()
    label_seg = nib.load(str(label_path)).get_data()
    image, label_seg = module.train_transforms(image, label_seg, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
    image = np.asarray(image)
    label_seg = np.asarray(label_seg)
    label = 0 if label_seg.sum()==0 else 1
    return {"image": image, "label_seg":label_seg, "label": label}

def generate_crop_data():
    config = Box.from_yaml(filename='/tmp2/tungi893610/template/configs/kits_crop_config.yaml')
    data_dir = Path(config.dataset.kwargs.data_dir)
    config.dataset.kwargs.update(data_dir=data_dir, type='train', task='seg')
    train_dataset = _get_instance(src.data.datasets, config.dataset)
    config.dataset.kwargs.update(data_dir=data_dir, type='valid', task='seg')
    valid_dataset = _get_instance(src.data.datasets, config.dataset)

    train_output_dir = '/tmp2/tungi893610/kits_crop_data/train'
    valid_output_dir = '/tmp2/tungi893610/kits_crop_data/valid'
    # Create output directory
    if not (Path(train_output_dir)).exists():
        Path(train_output_dir).mkdir(parents=True)
    if not (Path(valid_output_dir)).exists():
        Path(valid_output_dir).mkdir(parents=True)

    train_data_num = train_dataset.__len__()
    #train_trange = tqdm(range(train_data_num))
    valid_data_num = valid_dataset.__len__()
    #valid_trange = tqdm(range(valid_data_num))
    
    print('###generating training data...###')
    with tqdm(total=train_data_num) as pbar:
        for i in range(train_data_num):
            item = _get_item(train_dataset, i)
            image = item['image']
            seg = item['label_seg']
            label = item['label']
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(Path(train_output_dir) / f'imaging_{i}.nii.gz'))
            nib.save(nib.Nifti1Image(seg, np.eye(4)), str(Path(train_output_dir) / f'segmentation_{i}.nii.gz'))
            np.save(Path(train_output_dir) / f'classification_{i}.npy', label)
            pbar.update(1)

    print('###generating validation data...###')
    with tqdm(total=valid_data_num) as pbar:
        for i in range(valid_data_num):
            item = _get_item(valid_dataset, i)
            image = item['image']
            seg = item['label_seg']
            label = item['label']
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(Path(valid_output_dir) / f'imaging_{i}.nii.gz'))
            nib.save(nib.Nifti1Image(seg, np.eye(4)), str(Path(valid_output_dir) / f'segmentation_{i}.nii.gz'))
            np.save(Path(valid_output_dir) / f'classification_{i}.npy', label)
            pbar.update(1)
 
