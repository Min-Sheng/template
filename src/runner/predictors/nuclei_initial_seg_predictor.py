import csv
import json
import torch
import logging
import os.path
import numpy as np
from tqdm import tqdm
from pathlib import Path
import imageio
import torch.nn as nn

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

def load_img_name_json(data_split_list_path, data_type='train'):
    
    img_name_list = []
    with open(data_split_list_path, 'r') as file:
        
        data_list = json.load(file)
        img_name_list = data_list[data_type]
    
    return img_name_list

def to_img(arr):
    if arr.max() != 0:
        img = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    else:
        img = arr.astype(np.uint8)
    return img
    
class NucleiInitialSegPredictor(object):
    """The predictor for nuclei initial segmentation task.
    Args:
        data_dir (Path): The directory of the saved data.
        data_split_list (str): The path of the training and validation data split json file.
        transforms (list of Box): The preprocessing techniques applied to the data.
        device (torch.device): The device.
        net (BaseNet): The network architecture.
        metric_fns (list of torch.nn.Module): The metric functions.
        saved_dir (str): The directory to save the predicted videos, images and metrics (default: None).
        exported (bool): Whether to export the predicted video, images and metrics (default: False).
    """
    def __init__(self, data_dir, data_split_list, transforms, sample_size, shift, device, net, metric_fns, label_type, saved_dir=None, exported=None):
        self.data_dir = data_dir
        self.data_split_list = data_split_list
        self.transforms = compose(transforms)
        self.sample_size = sample_size
        self.shift = shift
        self.device = device
        self.net = net
        self.metric_fns = metric_fns
        self.label_type = label_type
        self.saved_dir = saved_dir
        self.exported = exported
        self.log = self._init_log()
    
    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        for metric in self.metric_fns:
            if metric.__class__.__name__ == 'Dice':
                for i in range(self.net.out_channels):
                    log[f'Dice_{i}'] = 0
            else:
                log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            metrics (list of torch.Tensor): The computed metrics.
        """
        for metric, _metric in zip(self.metric_fns, metrics):
            if metric.__class__.__name__ == 'Dice':
                for i, class_score in enumerate(_metric):
                    log[f'Dice_{i}'] += class_score.item()
            else:
                log[metric.__class__.__name__] += _metric.item()
    
    def predict(self):
        """The testing process.
        """
        self.net.eval()
        # Create the testing data path list
        data_paths = load_img_name_json(self.data_split_list, 'train')
        count = 0

        # Initital the list for saving metrics as a csv file
        header = ['name']
        for metric in self.metric_fns:
            if metric.__class__.__name__ == 'Dice':
                for i in range(self.net.out_channels):
                    header += [f'Dice_{i}']
            else:
                header += [metric.__class__.__name__]
        results = [header]
        
        if self.exported:
            csv_path = self.saved_dir / 'init_results.csv'
            output_dir = self.saved_dir / 'init_prediction'
            if not output_dir.is_dir():
                output_dir.mkdir(parents=True)
        
        trange = tqdm(data_paths, total=len(data_paths), desc='init_testing')
        for data_path in trange:
            image = imageio.imread(get_img_path(data_path, self.data_dir))
            image = np.asarray(image)
            if self.label_type=='instance':
                instance_label = np.load(get_instance_label_path(data_path, self.data_dir))
                full_label = (instance_label > 0)[...,None].astype(np.int32)
                label_dtype = torch.long
            elif self.label_type=='3cls_label':
                tricls_label = np.load(get_3cls_label_path(data_path, self.data_dir))
                full_label = tricls_label[...,None].astype(np.int32)
                label_dtype = torch.long
            elif self.label_type=='watershed_label':
                watershed_label = np.load(get_watershed_label_path(data_path, self.data_dir))
                full_label = watershed_label.astype(np.float32)
                label_dtype = torch.float
            
            data_shape = list(image.shape)
            image, full_label = self.transforms(image, full_label, normalize_tags=[True, False], dtypes=[torch.float, label_dtype])
            image, full_label = image.permute(2, 0, 1).contiguous().to(self.device), full_label.permute(2, 0, 1).contiguous().to(self.device)
            prediction = torch.zeros(1, self.net.out_channels, *image.shape[1:], dtype=torch.float32).to(self.device)
            pixel_count = torch.zeros(1, self.net.out_channels, *image.shape[1:], dtype=torch.float32).to(self.device)
            # Get the coordinated of each sampled volume
            starts, ends = [], []
            for j in range(0, data_shape[1], self.shift[1]):
                for i in range(0, data_shape[0], self.shift[0]):
                    ends.append([min(i+self.sample_size[0], data_shape[0]), \
                                 min(j+self.sample_size[1], data_shape[1])])
                    starts.append([max(ends[-1][0]-self.sample_size[0], 0), \
                                   max(ends[-1][1]-self.sample_size[1], 0)])

            # Get the prediction and calculate the average of the overlapped area
            for start, end in zip(starts, ends):
                input = image[:, start[0]:end[0], start[1]:end[1]]
                with torch.no_grad():
                    output = self.net(input.unsqueeze(dim=0))
                prediction[:, :, start[0]:end[0], start[1]:end[1]] += output
                pixel_count[:, :, start[0]:end[0], start[1]:end[1]] += 1.0
            prediction = prediction / pixel_count
            
            count += 1
            metrics = [metric(prediction, full_label.unsqueeze(dim=0)) for metric in self.metric_fns]
            self._update_log(self.log, metrics)
            
            # Export the prediction
            if self.exported:
                #filename = data_path.split('/')[-1]
                subdataset = data_path.split('/')[-2]
                prediction = prediction.argmax(dim=1).squeeze().cpu().numpy()
                prediction_dir = output_dir / data_path
                prediction_parent_dir = output_dir / subdataset
                if not prediction_parent_dir.is_dir():
                    prediction_parent_dir.mkdir(parents=True)
                imageio.imwrite(str(prediction_dir)+'.png', to_img(prediction))
                np.save(str(prediction_dir)+'.npy', prediction.astype(np.uint8))
                result = [data_path]
                for metric, _metric in zip(self.metric_fns, metrics):
                    if metric.__class__.__name__ == 'Dice':
                        for i, class_score in enumerate(_metric):
                            result.append(class_score.item())
                    else:
                        result.append(_metric.item())
                results.append([*result])

            dicts = {}
            for key, value in self.log.items():
                dicts[key] = f'{value / count: .3f}'
            trange.set_postfix(**dicts)

        for key in self.log:
            self.log[key] /= count

        if self.exported:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results)

        logging.info(f'Test log: {self.log}.')