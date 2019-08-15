import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from .base_logger import BaseLogger

def deNormalize(img, means=None, stds=None):
    if means != None and stds != None:
        means = torch.from_numpy(np.asarray(means))
        stds = torch.from_numpy(np.asarray(stds))
        img[:,0,:,:] = (img[:,0,:,:] * stds[0] + 1e-10) + means[0]
        img[:,1,:,:] = (img[:,1,:,:] * stds[1] + 1e-10) + means[1]
        img[:,2,:,:] = (img[:,2,:,:] * stds[2] + 1e-10) + means[2]
    return img

def disp_to_rgb(img):

    import matplotlib.colors
    import math
    
    device = img.device
    disp = img.detach()
    sigmoid = nn.Sigmoid()
    disp[:,2:4,:,:] = sigmoid(disp[:,2:4,:,:])
    disp = disp.cpu().permute(0,2,3,1).numpy()
    mask = disp[...,3] > 0.5
    
    norm = (np.sqrt(disp[..., 0] ** 2 + disp[..., 1] ** 2 )) + 1e-10
    disp[..., 0:2][mask] = (disp[..., 0:2] / norm[..., None])[mask]
    
    energy = np.zeros(disp[...,2].shape)
    energy[mask] = -disp[...,2][mask] + 1 + 1e-10

    disp[..., 0:2] = disp[..., 0:2] * (energy[...,None])
    a = (np.arctan2(disp[...,0], disp[...,1]) / math.pi + 1) / 2
    r = np.sqrt(disp[...,0] ** 2 + disp[...,1] ** 2)
    s = r / np.max(r)
    hsv_color = np.stack((a, s, np.ones_like(a)), axis=-1)
    rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)
    rgb_color[~mask] = 1
    disp[...,2][~mask] = 0

    rgb = [torch.from_numpy(rgb_color).permute(0,3,1,2).to(device), torch.from_numpy(mask[..., None]).permute(0,3,1,2).to(device)] 
    
    return tuple(rgb)

class NucleiInitialSegLogger(BaseLogger):
    """The logger for the nuclei initial segmentation task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict): The training batch.
            train_output (torch.Tensor): The training output.
            valid_batch (dict): The validation batch.
            valid_output (torch.Tensor): The validation output.
        """
        num_classes = train_output.size(1)
        
        
        if self.label_type=='watershed_label':
            
            train_img = make_grid(train_batch['ori_image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_watershed_semi_label, train_watershed_full_label = tuple(map(disp_to_rgb, [train_batch['semi_label'], train_batch['full_label']]))
            train_semi_label_disp = make_grid(train_watershed_semi_label[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_semi_label_mask = make_grid(train_watershed_semi_label[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_full_label_disp = make_grid(train_watershed_full_label[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            train_full_label_mask = make_grid(train_watershed_full_label[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            
            #train_semi_label_disp = make_grid(train_batch['semi_label'][:,0,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            #train_semi_label_mask = make_grid(train_batch['semi_label'][:,2,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            #train_full_label_disp = make_grid(train_batch['full_label'][:,0,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            #train_full_label_mask = make_grid(train_batch['full_label'][:,2,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            
            train_watershed_pred = disp_to_rgb(train_output)
            train_pred_disp = make_grid(train_watershed_pred[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_pred_mask = make_grid(train_watershed_pred[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            
            #train_pred_disp = make_grid(train_output[:,0,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            #train_pred_mask = make_grid(train_output[:,2,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            
            valid_img = make_grid(valid_batch['ori_image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_watershed_semi_label, valid_watershed_full_label = tuple(map(disp_to_rgb, [valid_batch['semi_label'], valid_batch['full_label']]))
            valid_semi_label_disp = make_grid(valid_watershed_semi_label[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_semi_label_mask = make_grid(valid_watershed_semi_label[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_full_label_disp = make_grid(valid_watershed_full_label[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            valid_full_label_mask = make_grid(valid_watershed_full_label[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            
            #valid_semi_label_disp = make_grid(valid_batch['semi_label'][:,0,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            #valid_semi_label_mask = make_grid(valid_batch['semi_label'][:,2,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            #valid_full_label_disp = make_grid(valid_batch['semi_label'][:,0,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 
            #valid_full_label_mask = make_grid(valid_batch['semi_label'][:,2,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1) 

            valid_watershed_pred = disp_to_rgb(valid_output)
            valid_pred_disp = make_grid(valid_watershed_pred[0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_pred_mask = make_grid(valid_watershed_pred[1].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            #valid_pred_disp = make_grid(valid_output[:,0,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
            #valid_pred_mask = make_grid(valid_output[:,2,:,:][:,None,:,:].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)

            train_grid = torch.cat((train_img, train_semi_label_disp, train_semi_label_mask, train_full_label_disp, train_full_label_mask, train_pred_disp, train_pred_mask), dim=-1)
            valid_grid = torch.cat((valid_img, valid_semi_label_disp, valid_semi_label_mask, valid_full_label_disp, valid_full_label_mask, valid_pred_disp, valid_pred_mask), dim=-1)

        else:
            train_img = make_grid(train_batch['ori_image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_semi_label = make_grid(train_batch['semi_label'].float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            train_full_label = make_grid(train_batch['full_label'].float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            train_pred = make_grid(train_output.argmax(dim=1, keepdim=True).float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            valid_img = make_grid(valid_batch['ori_image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_semi_label = make_grid(valid_batch['semi_label'].float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            valid_full_label = make_grid(valid_batch['full_label'].float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)
            valid_pred = make_grid(valid_output.argmax(dim=1, keepdim=True).float(), nrow=1, normalize=True, scale_each=True, range=(0, num_classes-1), pad_value=1)

            train_grid = torch.cat((train_img, train_semi_label, train_full_label, train_pred), dim=-1)
            valid_grid = torch.cat((valid_img, valid_semi_label, valid_full_label, valid_pred), dim=-1)
        
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)
