import torch
import torch.nn as nn

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
        output = output

        # Calculate the dice loss.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (output * target).sum(reduced_dims)
        union = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return 1 - score.mean()

class DiceWOBGLoss(nn.Module):
    """The Dice loss which ignores background.
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
        target = torch.zeros_like(output).scatter_(1, target, 1)[:, 1:, :, :]
        output = output[:, 1:, :, :]

        # Calculate the dice loss.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (output * target).sum(reduced_dims)
        union = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return 1 - score.mean()

class JaccardWOBGLoss(nn.Module):
    """The Jaccard (IoU) loss which ignores background.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            loss (torch.Tensor) (0): The Jaccard loss.
        """        
        # Get the one-hot encoding of the ground truth label.
        target = torch.zeros_like(output).scatter_(1, target, 1)[:, 1:, :, :]
        output = output[:, 1:, :, :]

        # Calculate the Jaccard loss.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = (output * target).sum(reduced_dims)
        union = (output).sum(reduced_dims) + (target).sum(reduced_dims) - intersection
        score = intersection / (union + 1e-10)
        return 1 - score.mean()

class BinaryCrossEntropyWOBGLoss(nn.Module):
    """The Binary Cross Entropy loss which ignores background.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        # Get the one-hot encoding of the ground truth label.
        target = torch.zeros_like(output).scatter_(1, target, 1)[:, 1:, :, :]
        output = output[:, 1:, :, :]
        
        # Calculate the binary cross entropy loss.
        BCELoss = nn.BCEWithLogitsLoss()

        return BCELoss(output, target)

class AngleLoss(nn.Module):
    """The angle loss for displacement vectors.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        
        output_disp = torch.clamp(output[:,0:2,:,:] * (1 / (torch.norm(output[:,0:2,:,:], p=2, dim=1, keepdim=True) + 1e-7)), -1+1e-7, 1-1e-7)
        target_disp = torch.clamp(target[:,0:2,:,:] * (1 / (torch.norm(target[:,0:2,:,:], p=2, dim=1, keepdim=True) + 1e-7)), -1+1e-7, 1-1e-7)
        
        # Calculate the angle loss
        errorAngles = torch.acos(torch.clamp((output_disp*target_disp*(target[:,2,:,:][:,None,:,:]>0).float()).sum(dim=1), -1+1e-7, 1-1e-7))
        angleLoss = torch.abs(errorAngles).mean()
        
        return angleLoss


class EnergyL2Loss(nn.Module):
    """The L2 loss for displacement energy.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        
        # Calculate the energy loss.
        output_energy = output[:,2,:,:]*target[:,3,:,:]
        target_energy = target[:,2,:,:]
        l2Loss = nn.MSELoss()
        energyLoss = l2Loss(output_energy, target_energy)

        return energyLoss

class BinaryDiceLoss(nn.Module):
    """The dice loss for binary mask.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        
        # Calculate the dice loss.
        num = output.size(0)
        output_mask = output[:,3,:,:].view(num, -1)
        target_mask = target[:,3,:,:].view(num, -1)

        intersection = 2.0 * (output_mask * target_mask).sum(1)
        union = (output_mask ** 2).sum(1) + (target_mask ** 2).sum(1)
        score = intersection / (union + 1e-10)
        diceLoss =  1 - score.mean()
    
        return diceLoss

class CenterMaskDiceLoss(nn.Module):
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
        output = output[:, 2:5, :, :]
        target = torch.zeros_like(output).scatter_(1, (target[:,2,:,:][:,None,:,:].long()), 1)
        output = output

        # Calculate the dice loss.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (output * target).sum(reduced_dims)
        union = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return 1 - score.mean()

class CenterMaskDiceWOBGLoss(nn.Module):
    """The Dice loss which ignores background.
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
        output = output[:, 2:5, :, :]
        target = torch.zeros_like(output).scatter_(1, (target[:,2,:,:][:,None,:,:].long()), 1)[:, 1:, :, :]
        output = output[:, 1:, :, :]

        # Calculate the dice loss.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (output * target).sum(reduced_dims)
        union = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return 1 - score.mean()