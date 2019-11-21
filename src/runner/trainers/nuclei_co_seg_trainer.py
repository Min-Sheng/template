import torch
import logging
import numpy as np
from tqdm import tqdm
import random
from src.model import DiceMeter, AverageValueMeter
from src.callbacks.scheduler import *
from src.utils import *

class NucleiCoSegTrainer(object):
    """The trainer for nuclei co segmentation task.
    """
    def __init__(self, device, train_dataloader, valid_dataloader, 
                 segmentators, sup_loss_fn, jsd_loss_fn, optimizers,
                 logger, monitor, num_epochs, label_type,
                 seg_schedulers, cot_scheduler , adv_scheduler, adv_training_dict):
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.segmentators = [segmentator.to(device) for segmentator in segmentators]
        self.S = len(self.segmentators)
        self.C = 3
        self.sup_loss_fn = sup_loss_fn.to(device)
        self.jsd_loss_fn = jsd_loss_fn.to(device)
        self.optimizers = optimizers
        self.label_type = label_type

        self.logger = logger
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.epoch = 1
        self.np_random_seeds = None

        # scheduler
        self.seg_schedulers = seg_schedulers
        self.cot_scheduler = cot_scheduler
        self.adv_scheduler = adv_scheduler
        self.adv_training_dict = adv_training_dict


    def train(self):
        """The training process.
        """
        if self.np_random_seeds is None:
            self.np_random_seeds = random.sample(range(10000000), k = self.num_epochs)

        while self.epoch <= self.num_epochs:
            # Reset the numpy random seed.
            np.random.seed(self.np_random_seeds[self.epoch - 1])

            # Do training and validation.
            logging.info(f'Epoch {self.epoch}.')
            train_log, train_batch, train_outputs = self._train_epoch()
            logging.info(f'Train log: {train_log}.')
            valid_log, valid_batch, valid_outputs = self._eval_epoch()
            logging.info(f'Valid log: {valid_log}.')

            # Adjust the learning rate.
            self.schedulerStep()

            # Record the log information and visualization.
            self.logger.write(self.epoch, train_log, train_batch, train_outputs,
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

    def _train_epoch(self):
        """Run an epoch for training.

        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """

        diceMeters = [DiceMeter(report_axises='all', method='2d', C=3) for _ in
                      range(self.S)]
        suplossMeters = [AverageValueMeter() for _ in range(self.S)]
        jsdlossMeter = AverageValueMeter()
        advlossMeter = AverageValueMeter()
        
        [segmentator.train() for segmentator in self.segmentators]
        dataloader = self.train_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc='training')
        log = self._init_train_log()
        for batch in trange:
            supervised_loss, jsd_loss, adv_loss = 0, 0, 0
            batch = self._allocate_data(batch)
            inputs, targets1, targets2 = self._get_inputs_targets(batch)
            outputs = list(map(lambda x: x(inputs), self.segmentators))
            sup_loss = list(map(lambda pred: self._compute_losses(pred, targets1, 'sup'), outputs))
            list(map(lambda x, y: x.add(y, targets2), diceMeters, outputs))
            list(map(lambda x, y: x.add(y.detach().data.cpu()), suplossMeters, sup_loss))
            supervised_loss = sum(sup_loss)

            map(lambda x: x.zero_grad(), self.optimizers)
            totalLoss = supervised_loss + 0.5 * jsd_loss + 0.5 * adv_loss
            totalLoss.backward()
            map(lambda x: x.step(), self.optimizers)
            
            batch_size = self.train_dataloader.batch_size
            self._update_train_log(log, diceMeters, suplossMeters, jsdlossMeter, advlossMeter)
            for i in range(self.S):
                if i == 0:
                    avg_log = log[f'S{i}']
                else:
                    for key, value in log[f'S{i}'].items():
                        avg_log[key] += value
            for key in avg_log:
                avg_log[key] /= self.S
            trange.set_postfix(**dict((key, f'{value: .3f}') for key, value in avg_log.items()))
        return log, batch, outputs

    def _eval_epoch(self):
        """Run an epoch for evaluation.

        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """
        diceMeters = [DiceMeter(report_axises='all', method='2d', C=3) for _ in
                      range(self.S)]
        vallossMeters = [AverageValueMeter() for _ in range(self.S)]

        [segmentator.eval() for segmentator in self.segmentators]
        dataloader = self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc='validation')
        log = self._init_eval_log()
        for batch in trange:
            supervised_loss, jsd_loss, adv_loss = 0, 0, 0
            batch = self._allocate_data(batch)
            inputs, targets1, targets2 = self._get_inputs_targets(batch)
            with torch.no_grad():
                outputs = list(map(lambda x: x(inputs), self.segmentators))
                sup_loss = list(map(lambda pred: self._compute_losses(pred, targets1, 'sup'), outputs))
                list(map(lambda x, y: x.add(y, targets2), diceMeters, outputs))
                list(map(lambda x, y: x.add(y.detach().data.cpu()), vallossMeters, sup_loss))

            batch_size = self.valid_dataloader.batch_size
            self._update_eval_log(log, diceMeters, vallossMeters)
            
            for i in range(self.S):
                if i == 0:
                    avg_log = log[f'S{i}']
                else:
                    for key, value in log[f'S{i}'].items():
                        avg_log[key] += value
            for key in avg_log:
                avg_log[key] /= self.S
            trange.set_postfix(**dict((key, f'{value: .3f}') for key, value in avg_log.items()))
        
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
    
    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.
        Returns:
            input (torch.Tensor): The data input.
            target1 (torch.LongTensor/torch.FloatTensor): The semi target.
            target2 (torch.LongTensor/torch.FloatTensor): The full target.
        """
        return batch['image'], batch['semi_label'], batch['full_label']

    def _compute_losses(self, output, target, mode):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.LongTensor/torch.FloatTensor): The semi target.
        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        if mode == 'sup':
            losses = self.sup_loss_fn(output, target)
        if mode == 'jsd':
            losses = self.jsd_loss_fn(output, target)
        return losses
    
    def _init_train_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {f'S{s}':{'Loss': 0, 
                'SupervisedLoss': 0, 
                'JsdLoss': 0, 
                'AdvLoss': 0, 
                'Dice': 0, 
                **{f'Dice_{c}': 0 for c in range(self.C)}} for s in range(self.S)}
        return log
    
    def _update_train_log(self, log, dice_meters, sup_loss_meters, jsd_loss_meter, adv_loss_meter):
        """Update the log.
        Args:
            log (dict): The log to be updated.
        """
        for s in range(self.S):
            log[f'S{s}']['Loss'] = sup_loss_meters[s].value()[0].cpu() # +
            log[f'S{s}']['SupervisedLoss'] = sup_loss_meters[s].value()[0].cpu()
            #log[f'S{s}']['JsdLoss'] = jsd_loss_meter.value()[0].cpu()
            #log[f'S{s}']['AdvLoss'] = adv_loss_meter.value()[0].cpu()
            log[f'S{s}']['Dice'] = dice_meters[s].value()[0][0].cpu()
            for c in range(self.C):
                log[f'S{s}'][f'Dice_{c}'] = dice_meters[s].value()[1][0][c].cpu()
    
    def _init_eval_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {f'S{s}':{'Loss': 0, 
                'SupervisedLoss': 0, 
                'Dice': 0, 
                **{f'Dice_{c}': 0 for c in range(self.C)}} for s in range(self.S)}
        return log
    
    def _update_eval_log(self, log, dice_meters, val_loss_meters):
        """Update the log.
        Args:
            log (dict): The log to be updated.
        """
        for s in range(self.S):
            log[f'S{s}']['Loss'] = val_loss_meters[s].value()[0].cpu()
            log[f'S{s}']['Dice'] = dice_meters[s].value()[0][0].cpu()
            for c in range(self.C):
                log[f'S{s}'][f'Dice_{c}'] = dice_meters[s].value()[1][0][c].cpu()
    
    def save(self, path):
        """Save the model checkpoint.
        Args:
            path (Path): The path to save the model checkpoint.
        """
        torch.save({
            'num_models': self.S,
            **{f'segmentator{i}': self.segmentators[i].state_dict() for i in range(self.S)},
            **{f'optimizer{i}': self.optimizers[i].state_dict() for i in range(self.S)},
            **{f'seg_schedulers{i}': self.seg_schedulers[i].state_dict() for i in range(self.S)},
            'monitor': self.monitor,
            'epoch': self.epoch,
            'random_state': random.getstate(),
            'np_random_seeds': self.np_random_seeds
        }, path)

    #def load(self, path):
    #    """Load the model checkpoint.
    #    Args:
    #        path (Path): The path to load the model checkpoint.
    #    """
    #    checkpoint = torch.load(path, map_location=self.device)
    #    self.net.load_state_dict(checkpoint['net'])
    #    self.optimizer.load_state_dict(checkpoint['optimizer'])
    #    if checkpoint['lr_scheduler']:
    #        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #    self.monitor = checkpoint['monitor']
    #    self.epoch = checkpoint['epoch'] + 1
    #    random.setstate(checkpoint['random_state'])
    #    self.np_random_seeds = checkpoint['np_random_seeds']
    
    def schedulerStep(self):
        if self.seg_schedulers == None:
            pass
        else:
            for seg_scheduler in self.seg_schedulers:
                if seg_scheduler == None:
                    pass
                else:
                    seg_scheduler.step()
        if self.cot_scheduler == None:
            pass
        else:
            self.cot_scheduler.step()
        if self.adv_scheduler == None:
            pass
        else:
            self.adv_scheduler.step()