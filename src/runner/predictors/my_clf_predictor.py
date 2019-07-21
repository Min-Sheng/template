import torch
from src.runner.predictors.base_predictor import BasePredictor


class ClfPredictor(BasePredictor):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.

        Returns:
            input (torch.Tensor): The data input.
            target (torch.LongTensor): The data target.
        """
        image = batch['image']
        image = torch.cat([image, image, image], dim=1) # Concatenate three one-channel images to a three-channels image.
        return image

    def _compute_losses(self, output, target):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.LongTensor): The data target.

        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = [loss(output, target) for loss in self.losses]
        return losses

    def _compute_metrics(self, output, target):
        """Compute the metrics.
        Args:
             output (torch.Tensor): The model output.
             target (torch.LongTensor): The data target.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        metrics = [metric(output, target) for metric in self.metrics]
        return metrics
