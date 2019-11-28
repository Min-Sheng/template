import argparse
import logging
import ipdb
import os
import sys
import torch
import random
import importlib
import yaml
from box import Box
from pathlib import Path

import src


def main(args):
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.main.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)

    logging.info(f'Save the config to "{config.main.saved_dir}".')
    with open(saved_dir / 'config.yaml', 'w+') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    if not args.test:
        # Make the experiment results deterministic.
        random.seed(config.main.random_seed)
        torch.manual_seed(random.getstate()[1][1])
        torch.cuda.manual_seed_all(random.getstate()[1][1])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logging.info('Create the device.')
        if 'cuda' in config.trainer.kwargs.device and not torch.cuda.is_available():
            raise ValueError("The cuda is not available. Please set the device in the trainer section to 'cpu'.")
        device = torch.device(config.trainer.kwargs.device)

        logging.info('Create the training and validation datasets.')
        config.dataset.kwargs.update(config.main.random_seed)
        data_dir = Path(config.dataset.kwargs.data_dir)
        config.dataset.kwargs.update(data_dir=data_dir, type='train')
        train_dataset = _get_instance(src.data.datasets, config.dataset)
        config.dataset.kwargs.update(data_dir=data_dir, type='val')
        valid_dataset = _get_instance(src.data.datasets, config.dataset)

        logging.info('Create the training and validation dataloaders.')
        cls = getattr(src.data.datasets, config.dataset.name)
        train_batch_size, valid_batch_size = config.dataloader.kwargs.pop('train_batch_size'), config.dataloader.kwargs.pop('valid_batch_size')
        config.dataloader.kwargs.update(collate_fn=getattr(cls, 'collate_fn', None), batch_size=train_batch_size)
        train_dataloader = _get_instance(src.data.dataloader, config.dataloader, train_dataset)
        config.dataloader.kwargs.update(batch_size=valid_batch_size)
        valid_dataloader = _get_instance(src.data.dataloader, config.dataloader, valid_dataset)
        
        label_type = config.dataset.kwargs.pop('label_type')
        config.segmentator.kwargs.update(label_type=label_type)

        logging.info('Create the network architecture.')
        num_models = config.main.num_models
        segmentators = [_get_instance(src.model.nets, config.segmentator) for i in range(num_models)]

        logging.info('Create the supervied loss functions.')
        defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
        if config.sup_loss.name in defaulted_loss_fns:
            sup_loss_fn = _get_instance(torch.nn, config.sup_loss)
        else:
            sup_loss_fn = _get_instance(src.model.losses, config.sup_loss)

        logging.info('Create the cot loss functions.')
        defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
        if config.cot_loss.name in defaulted_loss_fns:
            cot_loss_fn = _get_instance(torch.nn, config.cot_loss)
        else:
            cot_loss_fn = _get_instance(src.model.losses, config.cot_loss)

        logging.info('Create the optimizer.')
        optimizers = [_get_instance(torch.optim, config.optimizer, segmentators[i].parameters()) for i in range(num_models)]

        logging.info('Create the learning rate scheduler for each segmentor.')
        seg_schedulers = [_get_instance(torch.optim.lr_scheduler, config.seg_scheduler, optimizers[i]) for i in range(num_models)]

        logging.info('Create the learning rate scheduler for cotraining.')
        cot_scheduler = _get_instance(src.callbacks.scheduler, config.cot_scheduler) if config.get('cot_scheduler') else None

        logging.info('Create the learning rate scheduler for adversarial training.')
        adv_scheduler = _get_instance(src.callbacks.scheduler, config.adv_scheduler) if config.get('adv_scheduler') else None

        adv_training_dict = config.adv_training

        logging.info('Create the logger.')
        config.logger.kwargs.update(log_dir=saved_dir / 'log') 
        config.logger.kwargs.update(label_type=label_type) 
        logger = _get_instance(src.callbacks.loggers, config.logger)

        logging.info('Create the monitor.')
        config.monitor.kwargs.update(checkpoints_dir=saved_dir / 'checkpoints')
        config.monitor.kwargs.update(num_models=num_models) 
        monitor = _get_instance(src.callbacks.comonitor, config.monitor)
        
                
        logging.info('Create the trainer.')
        kwargs = {'device': device,
                  'train_dataloader': train_dataloader,
                  'valid_dataloader': valid_dataloader,
                  'segmentators': segmentators,
                  'sup_loss_fn': sup_loss_fn,
                  'cot_loss_fn': cot_loss_fn,
                  'optimizers': optimizers,
                  'seg_schedulers': seg_schedulers,
                  'cot_scheduler': cot_scheduler,
                  'adv_scheduler': adv_scheduler,
                  'adv_training_dict': adv_training_dict,
                  'logger': logger,
                  'monitor': monitor,
                  'label_type': label_type,
                  }
        config.trainer.kwargs.update(kwargs)
        trainer = _get_instance(src.runner.trainers, config.trainer)

        loaded_path = config.main.get('loaded_path')
        if loaded_path:
            logging.info(f'Load the previous checkpoint from "{loaded_path}".')
            trainer.load(Path(loaded_path))
            logging.info('Resume training.')
        else:
            logging.info('Start training.')
        trainer.train()
        logging.info('End training.')
    else:
        logging.info('Create the device.')
        if 'cuda' in config.predictor.kwargs.device and not torch.cuda.is_available():
            raise ValueError("The cuda is not available. Please set the device in the predictor section to 'cpu'.")
        device = torch.device(config.predictor.kwargs.device)

        logging.info('Create the testing dataset.')
        data_dir = Path(config.dataset.kwargs.data_dir)
        config.dataset.kwargs.update(data_dir=data_dir, type='test')
        test_dataset = _get_instance(src.data.datasets, config.dataset)

        logging.info('Create the testing dataloader.')
        test_dataloader = _get_instance(src.data.dataloader, config.dataloader, test_dataset)

        label_type = config.dataset.kwargs.pop('label_type')
        config.segmentator.kwargs.update(label_type=label_type)

        logging.info('Create the network architecture.')
        num_models = config.main.num_models
        segmentators = [_get_instance(src.model.nets, config.segmentator) for i in range(num_models)]

        logging.info('Create the supervied loss functions.')
        defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
        if config.sup_loss.name in defaulted_loss_fns:
            sup_loss_fn = _get_instance(torch.nn, config.sup_loss)
        else:
            sup_loss_fn = _get_instance(src.model.losses, config.sup_loss)

        logging.info('Create the predictor.')
        label_type = config.dataset.kwargs.pop('label_type')
        kwargs = {'device': device,
                  'test_dataloader': test_dataloader,
                  'segmentators': segmentators,
                  'sup_loss_fn': sup_loss_fn,
                  'label_type': label_type,
                  }
        config.predictor.kwargs.update(kwargs)
        predictor = _get_instance(src.runner.predictors, config.predictor)

        logging.info(f'Load the previous checkpoint from "{config.main.loaded_path}".')
        predictor.load(Path(config.main.loaded_path))
        logging.info('Start testing.')
        predictor.predict()
        logging.info('End testing.')


def _parse_args():
    parser = argparse.ArgumentParser(description="The script for the training and the testing.")
    parser.add_argument('config_path', type=Path, help='The path of the config file.')
    parser.add_argument('--test', action='store_true', help='Perform the training if specified; otherwise perform the testing.')
    args = parser.parse_args()
    return args


def _get_instance(module, config, *args):
    """
    Args:
        module (module): The python module.
        config (Box): The config to create the class object.

    Returns:
        instance (object): The class object defined in the module.
    """
    cls = getattr(module, config.name)
    kwargs = config.get('kwargs')
    return cls(*args, **config.kwargs) if kwargs else cls(*args)


if __name__ == "__main__":
    #with ipdb.launch_ipdb_on_exception():
    #    sys.breakpointhook = ipdb.set_trace
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
