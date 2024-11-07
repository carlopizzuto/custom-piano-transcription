import argparse
import os
import h5py

def check_hdf5s(hdf5s_dir):
    print(f"\nChecking HDF5 files in: {hdf5s_dir}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(hdf5s_dir):
        print(f"\nDirectory: {root}")
        print(f"Subdirectories: {dirs}")
        print(f"Files: {files}")
        
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                print(f"\nExamining: {file_path}")
                try:
                    with h5py.File(file_path, 'r') as hf:
                        print("Keys:", list(hf.keys()))
                        print("Attributes:", dict(hf.attrs))
                        print("Split:", hf.attrs['split'].decode())
                        print("Waveform shape:", hf['waveform'].shape)
                except Exception as e:
                    print(f"Error reading file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5s_dir', type=str, required=True)
    args = parser.parse_args()
    
    check_hdf5s(args.hdf5s_dir) 