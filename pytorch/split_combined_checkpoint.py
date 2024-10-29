import torch
import argparse
import os

def split_combined_checkpoint(args):
    """Split combined note and pedal model checkpoint into separate checkpoints."""
    
    combined_checkpoint_path = args.combined_checkpoint_path
    note_checkpoint_path = args.note_checkpoint_path
    pedal_checkpoint_path = args.pedal_checkpoint_path
    
    # Load the combined checkpoint
    checkpoint = torch.load(combined_checkpoint_path, map_location='cpu')
    
    # Check if the checkpoint contains 'model' key
    if 'model' in checkpoint:
        combined_model = checkpoint['model']
    else:
        combined_model = checkpoint
    
    # Extract note and pedal models
    note_model_state_dict = combined_model.get('note_model', None)
    pedal_model_state_dict = combined_model.get('pedal_model', None)
    
    if note_model_state_dict is None or pedal_model_state_dict is None:
        raise KeyError("The combined checkpoint does not contain both 'note_model' and 'pedal_model'.")
    
    # Save the note model checkpoint
    note_checkpoint = {'model': note_model_state_dict}
    torch.save(note_checkpoint, note_checkpoint_path)
    print(f"Note model saved to {note_checkpoint_path}")
    
    # Save the pedal model checkpoint
    pedal_checkpoint = {'model': pedal_model_state_dict}
    torch.save(pedal_checkpoint, pedal_checkpoint_path)
    print(f"Pedal model saved to {pedal_checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split combined checkpoint into separate note and pedal checkpoints.')
    parser.add_argument('--combined_checkpoint_path', type=str, required=True, help='Path to the combined checkpoint file.')
    parser.add_argument('--note_checkpoint_path', type=str, required=True, help='Output path for the note model checkpoint.')
    parser.add_argument('--pedal_checkpoint_path', type=str, required=True, help='Output path for the pedal model checkpoint.')
    
    args = parser.parse_args()
    split_combined_checkpoint(args)
