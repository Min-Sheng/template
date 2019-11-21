from functools import partial

import torch
import torch.nn.functional as F
from torch import einsum, Tensor

from .metricmeter import Metric
from src.utils.function_utils import one_hot, intersection, probs2one_hot, class2one_hot, simplex

# Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> float:
    assert label.shape == pred.shape # check if the shape is the same
    assert one_hot(label) # check if it is one hot
    assert one_hot(pred) # check if it is one hot

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


def toOneHot(pred_logit, mask):
    oh_predmask = probs2one_hot(F.softmax(pred_logit, 1))
    oh_mask = class2one_hot(mask.squeeze(1), pred_logit.shape[1])
    assert oh_predmask.shape == oh_mask.shape
    return oh_predmask, oh_mask


dice_coef = partial(meta_dice, "bcwh->bc")  # used for 2d dice
dice_batch = partial(meta_dice, "bcwh->c")  # used for 3d dice


class DiceMeter(Metric):
    def __init__(self, method='2d', report_axises='all', C=3) -> None:
        super().__init__()
        assert method in ('2d', '3d')
        assert report_axises == 'all' or isinstance(report_axises, list)
        self.method = method
        self.diceCall = dice_coef if self.method == '2d' else dice_batch
        self.report_axis = report_axises
        self.diceLog = []
        self.C = C

    def reset(self):
        self.diceLog = []

    def add(self, pred_logit, gt):
        dice_value = self.diceCall(*toOneHot(pred_logit, gt))
        if dice_value.shape.__len__() == 1:
            dice_value = dice_value.unsqueeze(0)
        assert dice_value.shape.__len__() == 2
        self.diceLog.append(dice_value)

    def value(self, **kwargs):
        log = self.log
        means = log.mean(0) # get mean over 0-axis elements
        stds = log.std(0) # get std over 0-axis elements
        report_means = log.mean(1) if self.report_axis == 'all' else log[:, self.report_axis].mean(1) # get means over 1-axis elements
        report_mean = report_means.mean() # get mean of the 1-axis means
        report_std = report_means.std() # get std of the 1-axis means
        
        return (report_mean, report_std), (means, stds)

    @property
    def log(self):
        try:
            log = torch.cat(self.diceLog)
        except:
            log = torch.Tensor([0 for _ in range(self.C)])
        if len(log.shape) == 1:
            log = log.unsqueeze(0)
        assert len(log.shape) == 2
        return log

    def detailed_summary(self) -> dict:
        _, (means, _) = self.value()
        return {f'DSC{i}': means[i].item() for i in range(len(means))}

    def summary(self) -> dict:
        (means, var), (_, _) = self.value()
        return {f'mDSC': means.item(),'mVars':var.item()}