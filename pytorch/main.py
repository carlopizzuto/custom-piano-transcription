import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import librosa
import logging
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.autograd

from utils.utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer, RegressionPostProcessor) 
from utils.data_generator import MaestroDataset, Augmentor, Sampler, TestSampler, collate_fn
from models import Regress_onset_offset_frame_velocity_CRNN, Regress_pedal_CRNN
from pytorch_utils import move_data_to_device
from losses import get_loss_func
from evaluate import SegmentEvaluator
import config
from model_utils import prepare_model_for_training


def train(args):
    """Train a piano transcription system.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Regressonset_regressoffset_frame_velocity_CRNN'
      loss_type: str, e.g. 'regress_onset_offset_frame_velocity_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      reduce_iteration: int
      resume_iteration: int
      early_stop: int
      device: 'cuda' | 'cpu'
      mini_data: bool
      checkpoint_path: str
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    if args.cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')  # Use Metal for M-series chips
        else:
            device = torch.device('cpu')
            print('Warning: CUDA/MPS not available, using CPU instead')
    else:
        device = torch.device('cpu')
    mini_data = args.mini_data
    filename = args.filename
    checkpoint_path = args.checkpoint_path
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    hop_seconds = config.hop_seconds
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 8

    # Loss function
    loss_func = get_loss_func(loss_type)

    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift),
        'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size))
    create_folder(logs_dir)

    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Model
    Model = eval(model_type)
    model = Model(frames_per_second=frames_per_second, classes_num=classes_num)
    
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled")
        print(f"Initial model device: {next(model.parameters()).device}")
    
    model = prepare_model_for_training(model)
    
    if checkpoint_path:
        print('Loading pretrained model from', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        if 'model' in checkpoint:  # Handle wrapped state dict
            checkpoint = checkpoint['model']
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print('Warning: Strict loading failed, trying non-strict loading')
            model.load_state_dict(checkpoint, strict=False)
    else:
        logging.info("Training from scratch.")

    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    else:
        raise Exception('Incorrect argumentation!')
    
    # Dataset
    train_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=max_note_shift, augmentor=augmentor)

    evaluate_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=0)

    # Sampler for training
    train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train', 
        segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # Sampler for evaluation
    evaluate_train_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='train', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    evaluate_validate_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='validation', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    evaluate_test_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='test', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = SegmentEvaluator(model, batch_size)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            model_type, 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
                '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    train_bgn_time = time.time()

    try:
        for batch_data_dict in train_loader:
            # Evaluation 
            if iteration % 5000 == 0:
                logging.info('------------------------------------')
                logging.info('Iteration: {}'.format(iteration))

            # Move data to device
            batch_data_dict = {
                key: move_data_to_device(value, device)
                for key, value in batch_data_dict.items()
            }
            
            # Training step
            model.train()
            optimizer.zero_grad()  # Zero gradients before forward pass
            
            with torch.autograd.detect_anomaly():
                # Forward pass
                batch_output_dict = model(batch_data_dict['waveform'])
                loss = loss_func(model, batch_output_dict, batch_data_dict)
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                optimizer.step()
            
            print(f"Iteration {iteration}, Loss: {loss.item()}")
            print(f"Device: {device}, Model device: {next(model.parameters()).device}")
            
            # Save model
            if iteration % 20000 == 0:
                checkpoint = {
                    'iteration': iteration, 
                    'model': model.module.state_dict(), 
                    'sampler': train_sampler.state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_iterations.pth'.format(iteration))
                    
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))
            
            # Reduce learning rate
            if iteration % reduce_iteration == 0 and iteration > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
            
            iteration += 1
            
            # Stop learning
            if iteration == early_stop:
                break
                
    except RuntimeError as e:
        print(f"Error during training: {str(e)}")
        print(f"Current device: {device}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Input shape: {batch_data_dict['waveform'].shape}")
        raise

    # Ensure model is in training mode
    print(f"Model parameters device check:")
    for name, param in model.named_parameters():
        if param.device != torch.device(device):
            print(f"Warning: {name} is on {param.device}, expected {device}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, required=True, choices=['none', 'aug'])
    parser_train.add_argument('--max_note_shift', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--checkpoint_path', type=str, required=False, help="Path to pretrained model checkpoint for fine-tuning.")
    parser_train.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser_train.add_argument('--detect_anomaly', action='store_true', default=False)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')