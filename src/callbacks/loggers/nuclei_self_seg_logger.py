import torch
import torch.nn as nn
import numpy as np
import matplotlib.colors
import math
from torchvision.utils import make_grid

from .base_logger import BaseLogger

def disp_to_rgb(img):
    device = img.device
    disp = img.detach()
    disp = disp.cpu().permute(0,2,3,1).numpy()
    if disp.shape[-1]==3:
        mask_w_c = disp[...,2]
        mask = mask_w_c > 0
    elif disp.shape[-1]==5:
        mask_w_c = np.argmax(disp[...,2:5], axis=-1)
        mask = mask_w_c > 0
    norm = (np.sqrt(disp[..., 0] ** 2 + disp[..., 1] ** 2 )) + 1e-10
    disp[..., 0:2][mask] = (disp[..., 0:2] / norm[..., None])[mask]
    
    #energy = np.zeros(disp[...,2].shape)
    #energy[mask] = -disp[...,2][mask] + 1 + 1e-10

    #disp[..., 0:2] = disp[..., 0:2] * (energy[...,None])
    a = (np.arctan2(disp[...,0], disp[...,1]) / math.pi + 1) / 2
    r = np.sqrt(disp[...,0] ** 2 + disp[...,1] ** 2)
    s = r / np.max(r)
    hsv_color = np.stack((a, s, np.ones_like(a)), axis=-1)
    rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)
    rgb_color[~mask] = 1
    disp[...,2][~mask] = 0

    rgb = [torch.from_numpy(rgb_color).permute(0,3,1,2).to(device), torch.from_numpy(mask_w_c[...,None]).permute(0,3,1,2).to(device)] 
    
    return tuple(rgb)

class NucleiSelfSegLogger(BaseLogger):
    """The logger for the nuclei initial segmentation task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def write(self, epoch, train_log, train_batch, train_pseudo_label, train_outputs, valid_log, valid_batch, valid_pseudo_label, valid_outputs):
        """Plot the network architecture and the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_log (dict): The training log information.
            train_batch (dict or sequence): The training batch.
            train_outputs (torch.Tensor or sequence of torch.Tensor): The training outputs.
            valid_log (dict): The validation log information.
            valid_batch (dict or sequence): The validation batch.
            valid_outputs (torch.Tensor or sequence of torch.Tensor): The validation outputs.
        """
        self._add_scalars(epoch, train_log, valid_log)
        self._add_images(epoch, train_batch, train_pseudo_label, train_outputs, valid_batch, valid_pseudo_label, valid_outputs)

    def _add_images(self, epoch, train_batch, train_pseudo_label, train_outputs, valid_batch, valid_pseudo_label, valid_outputs):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict): The training batch.
            train_output (torch.Tensor): The training output.
            valid_batch (dict): The validation batch.
            valid_output (torch.Tensor): The validation output.
        """
        num_classes = train_outputs.size(1)
        
        
        if self.label_type=='watershed_label':
            
            train_img = make_grid(train_batch['ori_image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_watershed_pseudo_label, train_watershed_full_label = tuple(map(disp_to_rgb, [train_pseudo_label, train_batch['full_label']]))
            train_pseudo_label_disp = make_grid(train_watershed_pseudo_label[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_pseudo_label_mask = make_grid(train_watershed_pseudo_label[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_full_label_disp = make_grid(train_watershed_full_label[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            train_full_label_mask = make_grid(train_watershed_full_label[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            
            train_watershed_pred = disp_to_rgb(train_outputs)
            train_pred_disp = make_grid(train_watershed_pred[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_pred_mask = make_grid(train_watershed_pred[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
                        
            valid_img = make_grid(valid_batch['ori_image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_watershed_pseudo_label, valid_watershed_full_label = tuple(map(disp_to_rgb, [valid_pseudo_label, valid_batch['full_label']]))
            valid_pseudo_label_disp = make_grid(valid_watershed_pseudo_label[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_pseudo_label_mask = make_grid(valid_watershed_pseudo_label[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_full_label_disp = make_grid(valid_watershed_full_label[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            valid_full_label_mask = make_grid(valid_watershed_full_label[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 

            valid_watershed_pred = disp_to_rgb(valid_outputs)
            valid_pred_disp = make_grid(valid_watershed_pred[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_pred_mask = make_grid(valid_watershed_pred[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)

            train_grid = torch.cat((train_img, train_pseudo_label_disp, train_pseudo_label_mask, train_full_label_disp, train_full_label_mask, train_pred_disp, train_pred_mask), dim=-1)
            valid_grid = torch.cat((valid_img, valid_pseudo_label_disp, valid_pseudo_label_mask, valid_full_label_disp, valid_full_label_mask, valid_pred_disp, valid_pred_mask), dim=-1)

        else:
            train_img = make_grid(train_batch['ori_image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_pseudo_label_mask = make_grid(train_pseudo_label.float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            train_full_label = make_grid(train_batch['full_label'].float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            train_pred = make_grid(train_outputs.argmax(dim=1, keepdim=True).float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            valid_img = make_grid(valid_batch['ori_image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_pseudo_label_mask = make_grid(valid_pseudo_label.float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            valid_full_label = make_grid(valid_batch['full_label'].float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            valid_pred = make_grid(valid_outputs.argmax(dim=1, keepdim=True).float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            train_grid = torch.cat((train_img, train_pseudo_label_mask, train_full_label, train_pred), dim=-1)
            valid_grid = torch.cat((valid_img, valid_pseudo_label_mask, valid_full_label, valid_pred), dim=-1)
        
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)
