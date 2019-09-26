import torch
import logging
import numpy as np
from tqdm import tqdm
import random

class NucleiSelfSegTrainer(object):
    """The trainer for nuclei self segmentation task.
    """
    def __init__(self, initial_model_path, device, train_dataloader, valid_dataloader,
                 label_generator, net, loss_fns, loss_weights, metric_fns, optimizer,
                 lr_scheduler, logger, monitor, num_rounds, num_epochs_per_rounds, label_type):
        self.initial_model_path = initial_model_path
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.label_generator =label_generator.to(device)
        self.net = net.to(device)
        self.loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.metric_fns = [metric_fn.to(device) for metric_fn in metric_fns]
        self.optimizer = optimizer
        self.label_type = label_type
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
            raise NotImplementedError('Do not support torch.optim.lr_scheduler.CyclicLR scheduler yet.')
        self.lr_scheduler = lr_scheduler

        self.logger = logger
        self.monitor = monitor
        self.num_rounds = num_rounds
        self.num_epochs_per_rounds = num_epochs_per_rounds

        self.epoch = 1
        self.np_random_seeds = None

    def train(self):
        """The training process.
        """
        if self.np_random_seeds is None:
            self.np_random_seeds = random.sample(range(10000000), k = self.num_epochs_per_rounds * self.num_rounds)
        print(self.initial_model_path)
        initial_model_checkpoint = torch.load(self.initial_model_path, map_location=self.device)
        self.net.load_state_dict(initial_model_checkpoint['net'])
        self.label_generator.load_state_dict(initial_model_checkpoint['net'])

        for rnd in range(0, self.num_rounds):
            # Load the model weights from the previous round
            if rnd > 0:
                self.label_generator.load_state_dict(self.net.state_dict())
            for epoch in range(0, self.num_epochs_per_rounds):
                # Reset the numpy random seed.
                np.random.seed(self.np_random_seeds[self.epoch - 1])

                # Do training and validation.
                logging.info(f'Round {rnd+1}.')
                logging.info(f'Epoch {epoch+1}.')
                train_log, train_batch, pseudo_label, train_outputs = self._run_epoch('training')
                logging.info(f'Train log: {train_log}.')
                valid_log, valid_batch, valid_outputs = self._run_epoch('validation')
                logging.info(f'Valid log: {valid_log}.')

                # Adjust the learning rate.
                if self.lr_scheduler is None:
                    pass
                elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and mode == 'validation':
                    self.lr_scheduler.step(valid_log['Loss'])
                else:
                    self.lr_scheduler.step()

                # Record the log information and visualization.
                self.logger.write(self.epoch, train_log, train_batch, pseudo_label, train_outputs,
                                valid_log, valid_batch, valid_outputs)

                # Save the regular checkpoint.
                saved_path = self.monitor.is_saved(self.epoch)
                if saved_path:
                    logging.info(f'Save the checkpoint to {saved_path}.')
                    self.save(saved_path)

                # Save the best checkpoint.
                saved_path = self.monitor.is_best(valid_log)
                if saved_path:
                    logging.info(f'Save the best checkpoint to {saved_path} ({self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')
                    self.save(saved_path)
                else:
                    logging.info(f'The best checkpoint is remained (at epoch {self.epoch - self.monitor.not_improved_count}, {self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')

                # Early stop.
                if self.monitor.is_early_stopped():
                    logging.info('Early stopped.')
                    break

                self.epoch +=1

        self.logger.close()

    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('training' or 'validation').

        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """
        self.label_generator.eval()
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            if mode == 'training':
                inputs, targets1, targets2 = self._get_inputs_targets(batch, mode)
                with torch.no_grad():
                    pseudo_label = self.label_generator(inputs)
                pseudo_label = pseudo_label.argmax(dim=1, keepdim=True)
                if self.label_type=='3cls_label':
                    new_pseudo_label = torch.where((targets1==1) | (pseudo_label==1), torch.full_like(targets1, 1), torch.zeros_like(targets1))
                    new_pseudo_label = torch.where((targets1==2) | (pseudo_label==2), torch.full_like(targets1, 2), new_pseudo_label)
                outputs = self.net(inputs)
                
                losses = self._compute_losses(outputs, new_pseudo_label)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                metrics =  self._compute_metrics(outputs, targets2)
            else:
                inputs, targets1, targets2 = self._get_inputs_targets(batch, mode)
                with torch.no_grad():
                    outputs = self.net(inputs)
                    losses = self._compute_losses(outputs, targets1)
                    loss = (torch.stack(losses) * self.loss_weights).sum()
                metrics =  self._compute_metrics(outputs, targets2)

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count
        if mode == 'training':
            return log, batch, new_pseudo_label, outputs
        else:
            return log, batch, outputs

    def _allocate_data(self, batch):
        """Allocate the data to the device.
        Args:
            batch (dict or sequence): A batch of the data.

        Returns:
            batch (dict or sequence): A batch of the allocated data.
        """
        if isinstance(batch, dict):
            return dict((key, self._allocate_data(data)) for key, data in batch.items())
        elif isinstance(batch, list):
            return list(self._allocate_data(data) for data in batch)
        elif isinstance(batch, tuple):
            return tuple(self._allocate_data(data) for data in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
    
    def _get_inputs_targets(self, batch, mode):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.
        Returns:
            input (torch.Tensor): The data input.
            target1 (torch.LongTensor/torch.FloatTensor): The semi target.
            target2 (torch.LongTensor/torch.FloatTensor): The full target.
        """
        return batch['image'], batch['semi_label'], batch['full_label']

    def _compute_losses(self, output, target):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.LongTensor/torch.FloatTensor): The semi target.
        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = [loss(output, target) for loss in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target):
        """Compute the metrics.
        Args:
             output (torch.Tensor): The model output.
             target (torch.LongTensor/torch.FloarTensor): The full target.
        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        metrics = [metric(output, target) for metric in self.metric_fns]
        return metrics

    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        log['Loss'] = 0
        for loss in self.loss_fns:
            log[loss.__class__.__name__] = 0
        for metric in self.metric_fns:
            if metric.__class__.__name__ == 'Dice':
                log['Dice'] = 0
                for i in range(self.net.out_channels):
                    log[f'Dice_{i}'] = 0
            elif metric.__class__.__name__ == 'CenterMaskDice':
                log['CenterMaskDice'] = 0
                for i in range(self.net.out_channels-2):
                    log[f'CenterMaskDice_{i}'] = 0
            else:
                log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, batch_size, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (list of torch.Tensor): The computed losses.
            metrics (list of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size
        for loss, _loss in zip(self.loss_fns, losses):
            log[loss.__class__.__name__] += _loss.item() * batch_size
        for metric, _metric in zip(self.metric_fns, metrics):
            if metric.__class__.__name__ == 'Dice':
                log['Dice'] += _metric.mean().item() * batch_size
                for i, class_score in enumerate(_metric):
                    log[f'Dice_{i}'] += class_score.item() * batch_size
            elif metric.__class__.__name__ == 'CenterMaskDice':
                log['CenterMaskDice'] += _metric.mean().item() * batch_size
                for i, class_score in enumerate(_metric):
                    log[f'CenterMaskDice_{i}'] += class_score.item() * batch_size
            else:
                log[metric.__class__.__name__] += _metric.item() * batch_size
    
    def save(self, path):
        """Save the model checkpoint.
        Args:
            path (Path): The path to save the model checkpoint.
        """
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'monitor': self.monitor,
            'epoch': self.epoch,
            'random_state': random.getstate(),
            'np_random_seeds': self.np_random_seeds
        }, path)

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.monitor = checkpoint['monitor']
        self.epoch = checkpoint['epoch'] + 1
        random.setstate(checkpoint['random_state'])
        self.np_random_seeds = checkpoint['np_random_seeds']
