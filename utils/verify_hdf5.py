import h5py
import os
import sys

def verify_hdf5(hdf5_path):
    print(f"\nVerifying HDF5 file: {hdf5_path}")
    try:
        with h5py.File(hdf5_path, 'r') as hf:
            print("Attributes:", dict(hf.attrs))
            print("Keys:", list(hf.keys()))
            print("Waveform shape:", hf['waveform'].shape if 'waveform' in hf else "No waveform")
            print("Split:", hf.attrs['split'].decode() if 'split' in hf.attrs else "No split")
    except Exception as e:
        print(f"Error reading file: {str(e)}")

if __name__ == '__main__':
    hdf5_dir = sys.argv[1]
    for root, _, files in os.walk(hdf5_dir):
        for file in files:
            if file.endswith('.h5'):
                verify_hdf5(os.path.join(root, file)) 