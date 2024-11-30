# utils/fix_midi.py
import os
from mido import MidiFile, MidiTrack, MetaMessage, Message

def scale_delta_time(delta, scaling_factor):
    """
    Scale the delta time with the given scaling factor.
    """
    return max(int(delta * scaling_factor), 0)

def convert_midi(midi_path, output_path, target_ticks_per_beat=480, target_tempo=500000):
    """
    Convert a single-track MIDI file to a dual-track MIDI file with scaled timing and updated metadata.

    Args:
        midi_path (str): Path to the original MIDI file.
        output_path (str): Path to save the converted MIDI file.
        target_ticks_per_beat (int, optional): Desired ticks per beat. Defaults to 480.
        target_tempo (int, optional): Desired tempo in microseconds per beat. Defaults to 500000.
    """
    # Read the original MIDI file
    original_midi = MidiFile(midi_path)
    original_ticks_per_beat = original_midi.ticks_per_beat

    # Calculate scaling factor
    scaling_factor = target_ticks_per_beat / original_ticks_per_beat

    # Create new MIDI file with target ticks_per_beat
    new_midi = MidiFile(ticks_per_beat=target_ticks_per_beat)

    # Create Track 0 with updated meta messages
    meta_track = MidiTrack()
    meta_track.append(MetaMessage('set_tempo', tempo=target_tempo, time=0))
    meta_track.append(MetaMessage('time_signature',
                                   numerator=4,
                                   denominator=4,
                                   clocks_per_click=24,
                                   notated_32nd_notes_per_beat=8,
                                   time=0))
    meta_track.append(MetaMessage('end_of_track', time=1))
    new_midi.tracks.append(meta_track)

    # Create Track 1 for MIDI events
    midi_events_track = MidiTrack()

    # Initialize previous absolute time
    absolute_time = 0

    for msg in original_midi.tracks[0]:
        # Skip original meta messages except 'set_tempo' and 'time_signature'
        if msg.is_meta and msg.type not in ['set_tempo', 'time_signature']:
            continue  # Skip other meta messages

        # Scale delta time
        scaled_delta = scale_delta_time(msg.time, scaling_factor)

        # Create a new message with scaled delta time
        if msg.is_meta:
            if msg.type == 'set_tempo':
                # Update tempo if you want to keep original tempo
                # To change tempo, use target_tempo
                new_msg = MetaMessage('set_tempo', tempo=target_tempo, time=scaled_delta)
            else:
                new_msg = msg.copy(time=scaled_delta)
        else:
            new_msg = msg.copy(time=scaled_delta)

        midi_events_track.append(new_msg)

    # End of Track
    midi_events_track.append(MetaMessage('end_of_track', time=1))
    new_midi.tracks.append(midi_events_track)

    # Save the new MIDI file
    new_midi.save(output_path)
    print(f"Converted {os.path.basename(midi_path)} to dual-track MIDI at {output_path}")

def process_dataset(midi_dir, output_dir, target_ticks_per_beat=480, target_tempo=500000):
    """
    Process all MIDI files in the dataset directory, converting them to dual-track MIDI files.

    Args:
        midi_dir (str): Directory containing original MIDI files.
        output_dir (str): Directory to save the converted MIDI files.
        target_ticks_per_beat (int, optional): Desired ticks per beat. Defaults to 480.
        target_tempo (int, optional): Desired tempo in microseconds per beat. Defaults to 500000.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(midi_dir):
        if filename.lower().endswith(('.midi', '.mid')):
            midi_path = os.path.join(midi_dir, filename)
            output_path = os.path.join(output_dir, filename)
            convert_midi(midi_path, output_path, target_ticks_per_beat, target_tempo)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert single-track MIDI files to dual-track MIDI files with scaled timing and updated metadata.")
    parser.add_argument('--midi_dir', type=str, required=True, help='Directory containing original MIDI files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the converted MIDI files.')
    parser.add_argument('--ticks_per_beat', type=int, default=480, help='Desired ticks per beat.')
    parser.add_argument('--tempo', type=int, default=500000, help='Desired tempo in microseconds per beat.')

    args = parser.parse_args()

    process_dataset(args.midi_dir, args.output_dir, args.ticks_per_beat, args.tempo)