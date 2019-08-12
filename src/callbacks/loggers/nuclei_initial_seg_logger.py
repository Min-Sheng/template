import torch
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
