import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from box import Box

class DiceLoss(nn.Module):
    """The Dice loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.

        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """
        # Get the one-hot encoding of the ground truth label.
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice loss.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (output * target).sum(reduced_dims)
        union = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return 1 - score.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.tensor(weight).cuda()
        #self.accumulate_loss = 0
        #self.total_num = 0

    #def _reset(self):
        #self.accumulate_loss = 0
        #self.total_num = 0

    def forward(self, output, target):
        loss_func = nn.CrossEntropyLoss(weight = self.weight)
        loss = loss_func(output, target)
        return loss



class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        # config = Box.from_yaml(filename='/home/extra/tungi893610/template/configs/kits_clf_config.yaml')
        self.alpha = Variable(torch.FloatTensor(alpha))
        self.gamma = gamma

    def forward(self, output, target):
        print(output.size(0))
        print(output.size(1))
        print(output.size(2))
        N = output.size(0)
        C = output.size(1)
        P = F.softmax(output)

        class_mask = output.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = target.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if output.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        loss = batch_loss.sum()
        return loss

