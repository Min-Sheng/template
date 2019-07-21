import csv
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class PredClfSegDataset(BaseDataset):
    """The Kidney Tumor Segmentation (KiTS) Challenge dataset (ref: https://kits19.grand-challenge.org/) for the two-staged method.
    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        task (str): The task name ('clf' or 'seg').
        train_transforms (Box): The preprocessing and augmentation techiques applied to the training data.
        valid_transforms (Box): The preprocessing and augmentation techiques applied to the validation data.
    """
    def __init__(self, data_split_csv, task, transforms, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.task = task
        self.transforms = compose(transforms) # for validation images
        self.clf_label_transforms = compose() # for 'clf' labels
        self.data_paths = []

        if self.type=='test':
            print(self.data_dir)
            self.data_paths = sorted(list(self.data_dir.glob('imaging*.nii.gz')))
            
        else:
            # Collect the data paths according to the dataset_split_csv.
            with open(self.data_split_csv, "r") as f:
                rows = csv.reader(f)
                for case_name, split_type in rows:
                    _image_paths = sorted(list((self.data_dir / case_name).glob('imaging*.nii.gz')))
                    _clf_label_paths = sorted(list((self.data_dir / case_name).glob('classification*.npy')))
                    _seg_label_paths = sorted(list((self.data_dir / case_name).glob('segmentation*.nii.gz')))
                    
                    if self.task == 'clf':
                        self.data_paths.extend([(image_path, seg_label_path) for image_path, seg_label_path \
                                in zip(_image_paths, _seg_label_paths)])
                    elif self.task == 'seg':
                        # Exclude the slice that does not contain the kidney or the tumer (foreground).
                        self.data_paths.extend([(image_path, seg_label_path) for image_path, clf_label_path, seg_label_path \
                                in zip(_image_paths, _clf_label_paths, _seg_label_paths) if np.load(clf_label_path) != 0])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        if self.type=='test':
            image = nib.load(str(self.data_paths[index])).get_data()
            image = self.transforms(image, normalize_tags=[True], dtypes=[torch.float])
            image = image.permute(2, 0, 1).contiguous()
            image_name = str(self.data_paths[index])
            return {"image": image}
        else:
            image_path, label_path = self.data_paths[index]
            image = nib.load(str(image_path)).get_data()
            seg_label = nib.load(str(label_path)).get_data()
            if self.task == 'clf':
                image, seg_label = self.transforms(image, seg_label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
                image = image.permute(2, 0, 1).contiguous()
                seg_label = np.asarray(seg_label)
                label = np.array(0) if seg_label.sum()==0 else np.array(1)
                label = self.clf_label_transforms(label, dtypes=[torch.long])
            elif self.task == 'seg':
                image, label = self.transforms(image, seg_label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
                image, label = image.permute(2, 0, 1).contiguous(), label.permute(2, 0, 1).contiguous()
            return {"image": image, "label": label}
