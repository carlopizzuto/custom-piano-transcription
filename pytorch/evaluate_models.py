import os
import sys
import argparse
import torch
from evaluate import SegmentEvaluator
from pytorch_utils import move_data_to_device
from utils.data_generator import MaestroDataset, collate_fn
from torch.utils.data import DataLoader
import config

def load_model(model_type, checkpoint_path, device):
    """Load the model from the checkpoint."""
    Model = eval(model_type)
    model = Model(frames_per_second=config.frames_per_second, classes_num=config.classes_num)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, batch_size, device):
    """Evaluate the model using SegmentEvaluator."""
    evaluator = SegmentEvaluator(model, batch_size)
    statistics = evaluator.evaluate(dataloader)
    return statistics

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
    
    # Load datasets
    evaluate_dataset = MaestroDataset(
        hdf5s_dir=os.path.join(args.workspace, 'hdf5s', 'maestro'),
        segment_seconds=config.segment_seconds,
        frames_per_second=config.frames_per_second,
        max_note_shift=0
    )
    
    evaluate_sampler = torch.utils.data.SequentialSampler(evaluate_dataset)
    evaluate_loader = DataLoader(
        dataset=evaluate_dataset,
        batch_size=args.batch_size,
        sampler=evaluate_sampler,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )
    
    # Load Initial Model
    initial_model = load_model(args.model_type, args.initial_checkpoint, device)
    initial_stats = evaluate_model(initial_model, evaluate_loader, args.batch_size, device)
    print(f'Initial Model Statistics: {initial_stats}')
    
    # Load Best Saved Model
    best_model = load_model(args.model_type, args.best_checkpoint, device)
    best_stats = evaluate_model(best_model, evaluate_loader, args.batch_size, device)
    print(f'Best Saved Model Statistics: {best_stats}')
    
    # Compare Metrics
    for key in initial_stats:
        print(f'--- Metric: {key} ---')
        print(f'Initial Model: {initial_stats[key]}')
        print(f'Best Model: {best_stats.get(key, "N/A")}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Initial and Best Models')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace')
    parser.add_argument('--model_type', type=str, required=True, help='Type of the model (e.g., Regress_onset_offset_frame_velocity_CRNN)')
    parser.add_argument('--initial_checkpoint', type=str, required=True, help='Path to the initial model checkpoint')
    parser.add_argument('--best_checkpoint', type=str, required=True, help='Path to the best saved model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA for evaluation')
    
    args = parser.parse_args()
    main(args)