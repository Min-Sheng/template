import torch
import torch.nn as nn


class Dice(nn.Module):
    """The Dice score.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.

        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        pred = output.argmax(dim=1, keepdim=True)
        pred = torch.zeros_like(output).scatter_(1, pred, 1)
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice score.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (pred * target).sum(reduced_dims)
        union = pred.sum(reduced_dims) + target.sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return score.mean(dim=0)


class Accuracy(nn.Module):
    """The accuracy for the classification task.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C): The model output.
            target (torch.LongTensor) (N): The data target.

        Returns:
            metric (torch.Tensor) (0): The accuracy.
        """
        pred = torch.argmax(output, dim=1)
        return (pred == target).float().mean()


class F1score(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        pred = torch.argmax(output, dim=1)
        pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
        tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
        TP = (pre_mask[:, 1]*tar_mask[:, 1]).float().sum()
        FP = (pre_mask[:, 1]*tar_mask[:, 0]).float().sum()
        FN = (pre_mask[:, 0]*tar_mask[:, 1]).float().sum()
        TN = (pre_mask[:, 0]*tar_mask[:, 0]).float().sum()
        #TP = ((pred==1) & (target==1)).float().sum()
        #TN = ((pred==0) & (target==0)).float().sum()
        #FN = ((pred==0) & (target==1)).float().sum()
        #FP = ((pred==1) & (target==0)).float().sum()
        precision = TP/((TP+FP) + 1e-10) 
        recall = TP/((TP+FN) + 1e-10)
        F1 = 2*precision*recall / ((precision+recall) + 1e-10)
        return F1

class TPRate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        pred = torch.argmax(output, dim=1) 
        pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
        tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
        TP = (pre_mask[:, 1]*tar_mask[:, 1]).float().sum()
        FN = (pre_mask[:, 0]*tar_mask[:, 1]).float().sum()
        return TP/(TP+FN+1e-10)

class FPRate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        pred = torch.argmax(output, dim=1) 
        pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
        tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
        FP = (pre_mask[:, 1]*tar_mask[:, 0]).float().sum()
        TN = (pre_mask[:, 0]*tar_mask[:, 0]).float().sum()
        return FP/(FP+TN+1e-10)

class TNRate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        pred = torch.argmax(output, dim=1) 
        pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
        tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
        FP = (pre_mask[:, 1]*tar_mask[:, 0]).float().sum()
        TN = (pre_mask[:, 0]*tar_mask[:, 0]).float().sum()
        return TN/(FP+TN+1e-10)


class FNRate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        pred = torch.argmax(output, dim=1) 
        pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
        tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
        TP = (pre_mask[:, 1]*tar_mask[:, 1]).float().sum()
        FN = (pre_mask[:, 0]*tar_mask[:, 1]).float().sum()
        return FN/(TP+FN+1e-10)






