import torch
import torch.nn as nn
import numpy as np
import numexpr as ne
from sklearn.metrics import f1_score

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
    
class CenterMaskDice(nn.Module):
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
            loss (torch.Tensor) (0): The dice loss.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        output = output[:, 2:5, :, :]
        target = target[:,2,:,:][:,None,:,:].long()
        pred = output.argmax(dim=1, keepdim=True)
        pred = torch.zeros_like(output).scatter_(1, pred, 1)
        target = torch.zeros_like(output).scatter_(1, target, 1)
        
        # Calculate the dice score.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (pred * target).sum(reduced_dims)
        union = pred.sum(reduced_dims) + target.sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return score.mean(dim=0)

class Dummy(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.Tensor([0])

class AggreagteJaccardIndex(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def compute_intersect_union(self, m, pred, pred_mark_isused, idx_pred):
        # check the prediction has been used or not
        if pred_mark_isused[idx_pred]:
            intersect = 0
            union = np.count_nonzero(m)
        else:
            p = (pred == idx_pred)
            # replace multiply with bool operation
            s = ne.evaluate("m&p")
            intersect = np.count_nonzero(s)
            #intersect = np.count_nonzero(m & p)
            #u1 = np.count_nonzero(m)
            #u2 = np.count_nonzero(p)
            #union = ne.evaluate("u1+u2-intersect")
            union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
        return (intersect, union)
    
    # fast version of Aggregated Jaccrd Index
    def agg_jc_index(self, pred, mask):
        """Calculate aggregated jaccard index for prediction & GT mask
        reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0

        mask: Ground truth mask, shape = [1000, 1000, instances]
        pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance

        Returns: Aggregated Jaccard index for GT & mask 
        """
        
        mask=mask.astype(np.bool)
        c = 0 # count intersection
        u = 0 # count union
        #tqdm.monitor_interval = 0 # disable tqdm monitor to prevent warning message
        pred_instance = pred.max() # predcition instance number
        if pred_instance==0:
            return 0
        pred_mark_used = [] # mask used
        pred_mark_isused = np.zeros((pred_instance+1), dtype=bool)
        
        #for idx_m in tqdm_notebook(range(len(mask[0,0,:]))):
        for idx_m in range(len(mask[0,0,:])):
            # m = mask[:,:,idx_m]
            m = np.take(mask, idx_m, axis=2)
            #intersect_list = []
            #union_list = []
            #iou_list = []
            
            intersect_list, union_list = zip(*[self.compute_intersect_union(m, pred, pred_mark_isused, idx_pred) for idx_pred in range(1, pred_instance+1)])
            
            #print(intersect_list)
            """
            for idx_pred in range(1, pred_instance+1):
                # check the prediction has been used or not
                if pred_mark_isused[idx_pred] == True:
                    intersect = 0
                    union = np.count_nonzero(m)
                else:
                    p = (pred == idx_pred)
                    
                    # replace multiply with bool operation 
                    s = ne.evaluate("m&p")
                    intersect = np.count_nonzero(s)
                    union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
                
                intersect_list.append(intersect)
                union_list.append(union)
                #print(intersect_list)
            """
            iou_list = np.array(intersect_list) / np.array(union_list)    
            hit_idx = np.argmax(iou_list)
            c += intersect_list[hit_idx]
            u += union_list[hit_idx]
            pred_mark_used.append(hit_idx)
            pred_mark_isused[hit_idx+1] = True
            
        pred_mark_used = [x+1 for x in pred_mark_used]
        pred_fp = set(np.unique(pred)) - {0} - set(pred_mark_used)
        pred_fp_pixel = np.sum([np.sum(pred==i) for i in pred_fp])

        u += pred_fp_pixel
        #print (c / u)
        return (c / u)
    def forward(self, output, target):
        """
        Args:
            output (numpy.array) (H, W, K)
            target (numpy.array) (H, W)
        Returns:
            metric (torch.Tensor): The dice scores for each class.
        """
        return torch.Tensor([self.agg_jc_index(output, target)])

class F1Score(nn.Module):
    """The F1 score.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (numpy.array) (H, W)
            target (numpy.array) (H, W)
        Returns:
            metric (torch.Tensor): The dice scores for each class.
        """
        
        return f1_score((target>0).flatten(), (output>0).flatten())
