
import csv
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class KitsClfSegDataset(BaseDataset):       
    def __init__(self, data_split_csv, task, train_transforms, valid_transforms, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.task = task
        self.train_transforms = compose(train_transforms) # for training images and 'seg' labels
        self.valid_transforms = compose(valid_transforms) # for validation images
        self.clf_label_transforms = compose() # for 'clf' labels
        self.data_paths = []

        # Collect the data paths 
        type_ = 'Training' if self.type == 'train' else 'Validation'
        if type_=='Training':
            dir_path = Path('/tmp2/tungi893610/kits_crop_data/train')
            _image_paths = sorted(list(dir_path.glob('imaging*.nii.gz')))
            _clf_label_paths = sorted(list(dir_path.glob('classification*.npy')))
            _seg_label_paths = sorted(list(dir_path.glob('segmentation*.nii.gz')))
        else:
            dir_path = Path('/tmp2/tungi893610/kits_crop_data/valid')
            _image_paths = sorted(list(dir_path.glob('imaging*.nii.gz')))
            _clf_label_paths = sorted(list(dir_path.glob('classification*.npy')))
            _seg_label_paths = sorted(list(dir_path.glob('segmentation*.nii.gz')))
        if self.task == 'clf':
            self.data_paths.extend([(image_path, clf_label_path) for image_path, clf_label_path in zip(_image_paths, _clf_label_paths)]) 
        elif self.task == 'seg':
            # Exclude the slice that does not contain the kidney or the tumer (foreground).
            self.data_paths.extend([(image_path, seg_label_path) for image_path, clf_label_path, seg_label_path in zip(_image_paths, _clf_label_paths, _seg_label_paths) if np.load(clf_label_path) != 0])
    
        # print(type_ + str(len(self.data_paths)))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, label_path = self.data_paths[index]
        image = nib.load(str(image_path)).get_data()
        label = np.load(label_path) if self.task == 'clf' else nib.load(str(label_path)).get_data()
        if self.type == 'train' and self.task == 'clf':
            image = self.train_transforms(image)
            label = self.clf_label_transforms(label, dtypes=[torch.long])
            image = image.permute(2, 0, 1).contiguous()
        elif self.type == 'valid' and self.task == 'clf':
            image = self.valid_transforms(image)
            label = self.clf_label_transforms(label, dtypes=[torch.long])
            image = image.permute(2, 0, 1).contiguous()
        elif self.type == 'train' and self.task == 'seg':
            image, label = self.train_transforms(image, label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
            image, label = image.permute(2, 0, 1).contiguous(), label.permute(2, 0, 1).contiguous()
        elif self.type == 'valid' and self.task == 'seg':
            image, label = self.valid_transforms(image, label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
            image, label = image.permute(2, 0, 1).contiguous(), label.permute(2, 0, 1).contiguous()
        return {"image": image, "label": label}
