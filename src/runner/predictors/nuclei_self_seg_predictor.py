import cv2
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
from skimage.feature import peak_local_max
from skimage import measure
from skimage.morphology import watershed
from skimage.segmentation import random_walker

def get_img_path(img_name, nuclei_root):
    name = img_name.split('/')[-1]
    return os.path.join(nuclei_root, img_name, 'image', name + '.png')

def get_instance_label_path(img_name, nuclei_root):
    name = img_name.split('/')[-1]
    return os.path.join(nuclei_root, img_name, 'label', name + '_label.npy')

def get_no_overlap_label_path(img_name, nuclei_root):
    name = img_name.split('/')[-1]
    return os.path.join(nuclei_root, img_name, 'no_overlap_label', name + '_no_overlap_label.npy')
    
def get_3cls_label_path(img_name, nuclei_root):
    name = img_name.split('/')[-1]
    return os.path.join(nuclei_root, img_name, '3cls_label', name + '_3cls_label.npy')

def get_watershed_label_path(img_name, nuclei_root):
    name = img_name.split('/')[-1]
    return os.path.join(nuclei_root, img_name, 'watershed_label', name + '_watershed_label.npy')

def load_img_name_json(data_split_list_path, data_type='test'):
    
    img_name_list = []
    with open(data_split_list_path, 'r') as file:
        
        data_list = json.load(file)
        img_name_list = data_list[data_type]
    
    return sorted(img_name_list)

def to_img(arr):
    if arr.max() != 0:
        img = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    else:
        img = arr.astype(np.uint8)
    return img
    
class NucleiSelfSegPredictor(object):
    """The predictor for nuclei self-training segmentation task.
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
    def __init__(self, data_dir, data_split_list, transforms, sample_size, shift, device, net, metric_fns, label_type, post_process="watershed", saved_dir=None, exported=None):
        self.data_dir = data_dir
        self.data_split_list = data_split_list
        self.transforms = compose(transforms)
        self.sample_size = sample_size
        self.shift = shift
        self.device = device
        self.net = net
        self.metric_fns = metric_fns
        self.label_type = label_type
        self.post_process = post_process
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
        data_paths = load_img_name_json(self.data_split_list, 'test')
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
            if self.post_process=='watershed':
                csv_path = self.saved_dir / 'selftraining_results' / 'selftraining_results_ws.csv'
                output_dir = self.saved_dir / 'selftraining_results' / 'selftraining_prediction_ws'
            elif self.post_process=='randomwalk':
                csv_path = self.saved_dir / 'selftraining_results' / 'selftraining_results_rw.csv'
                output_dir = self.saved_dir / 'selftraining_results' / 'selftraining_prediction_rw'
            if not output_dir.is_dir():
                output_dir.mkdir(parents=True)
        
        trange = tqdm(data_paths, total=len(data_paths), desc='self-training_testing')
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
            
            no_overlap_label = np.load(get_no_overlap_label_path(data_path, self.data_dir))

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

            if self.label_type != 'watershed_label':
                prediction_np = prediction.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
                prediction_inner = prediction_np.copy()
                prediction_inner[np.isin(prediction_inner, 1)]=0
                prediction_local_max = peak_local_max(prediction_inner, indices=False, min_distance=3, exclude_border=False)
                prediction_markers = measure.label(prediction_local_max, connectivity=1, background=0)
                
                if self.post_process == 'watershed':
                    prediction_final = watershed(-prediction_np, prediction_markers, mask=prediction_np>0)
                elif self.post_process == 'randomwalk':
                    prediction_markers[np.where(prediction_np==0)] = -1
                    prediction_final = random_walker(prediction_np>0, prediction_markers, beta=10)
                    prediction_final[np.where(prediction_final==-1)]=0
                    prediction_final = measure.label(prediction_final, connectivity=1, background=0)
                
                #kernel = np.ones((11, 11),np.uint8)
                #prediction_final_instance=[]
                #for i in range(1, prediction_final.max()+1):
                #    temp = np.zeros_like(prediction_final, dtype=np.float64)
                #    temp[np.where(prediction_final==i)]=1
                #    temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
                #    prediction_final_instance.append(temp)
                #prediction_final_instance = np.dstack(prediction_final_instance)

                #prediction_final = np.zeros_like(prediction_final)
                #for i in range(prediction_final_instance.shape[-1]):
                #    prediction_final[np.where(prediction_final_instance[:, :, i])] = i+1
                
            else:
                prediction_np = prediction.squeeze().cpu().numpy().transpose(1,2,0)
                prediction_mask = np.argmax(prediction_np[..., 2:5], axis=-1).astype(np.uint8)
                prediction_disp = np.tanh(prediction_np[..., 0:2] * (prediction_mask>0)[..., None])
                strength = (np.sqrt(prediction_disp[..., 0]**2 + prediction_disp[..., 1]**2)) + 1e-10
                prediction_disp[..., 0:2][prediction_mask > 0] = (prediction_disp[..., 0:2]/strength[..., None])[prediction_mask > 0]
                prediction_np = np.dstack((prediction_disp, prediction_mask))

                prediction_local_max = peak_local_max(prediction_mask>1, indices=False, min_distance=3, exclude_border=False)
                prediction_markers = measure.label(prediction_local_max, connectivity=1, background=0)

                if self.post_process == 'watershed':
                    raise("No watershed post-processing method")
                elif self.post_process == 'randomwalk':
                    prediction_markers[np.where(prediction_disp[..., 1]==0)] = -1
                    prediction_final = random_walker(prediction_disp[..., 1], prediction_markers, beta = 100)
                    prediction_final[np.where(prediction_final==-1)]=0
                    prediction_final = measure.label(prediction_final, connectivity=1, background=0)

                    prediction_final[np.where(prediction_disp[..., 0]==0)] = -1
                    prediction_final = random_walker(prediction_disp[..., 0], prediction_final, beta = 100)
                    prediction_final[np.where(prediction_final==-1)]=0
                    prediction_final = measure.label(prediction_final, connectivity=1, background=0)
            
            no_overlap_label_instance=[]
            for i in range(1, no_overlap_label.max()+1):
                temp = np.zeros_like(no_overlap_label)
                temp[np.where(no_overlap_label==i)]=True
                no_overlap_label_instance.append(temp)
            no_overlap_label_instance = np.dstack(no_overlap_label_instance)

            metrics = []
            for metric in self.metric_fns:
                if metric.__class__.__name__ == 'AggreagteJaccardIndex':
                    metrics.append(metric(prediction_final, no_overlap_label_instance))
                elif metric.__class__.__name__ == 'F1Score':
                    metrics.append(metric(prediction_final, no_overlap_label))
                elif metric.__class__.__name__ == 'EnsembleDice':
                    metrics.append(metric(torch.ByteTensor(prediction_final.astype(np.int32)), torch.ByteTensor(no_overlap_label.astype(np.int32))))
                else:
                    metrics.append(metric(prediction, full_label.unsqueeze(dim=0)))
            
            self._update_log(self.log, metrics)
            
            # Export the prediction
            if self.exported:
                #filename = data_path.split('/')[-1]
                subdataset = data_path.split('/')[-2]
                prediction_dir = output_dir / data_path
                prediction_parent_dir = output_dir / subdataset
                if not prediction_parent_dir.is_dir():
                    prediction_parent_dir.mkdir(parents=True)
                if self.label_type == '3cls_label':
                    imageio.imwrite(str(prediction_dir)+'_pred_3cls.png', to_img(prediction_np))
                    np.save(str(prediction_dir)+'_pred_3cls.npy', prediction_np.astype(np.uint8))
                    np.save(str(prediction_dir)+'_pred_instance.npy', prediction_final.astype(np.int32))
                    np.save(str(prediction_dir)+'_gt_instance.npy', no_overlap_label.astype(np.int32))
                elif self.label_type == 'watershed_label':
                    np.save(str(prediction_dir)+'_pred_disp.npy', prediction_np.astype(np.uint8))
                    np.save(str(prediction_dir)+'_pred_instance.npy', prediction_final.astype(np.int32))
                    np.save(str(prediction_dir)+'_gt_instance.npy', no_overlap_label.astype(np.int32))
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