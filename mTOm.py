import pandas as pd
import os

def split_dir(dir, df, train_thresh=25):
    instrument = dir.split('_')[1]
    new_dir = os.path.join(dir, instrument, "MUS")
    
    i = 0
    for file in os.listdir(new_dir):
        if file.endswith('.txt'):
            base_name = file.replace('.txt', '')
            wav_path = base_name + '.wav'
            midi_path = base_name + '.mid'
            text_path = file
            song_name = base_name.replace('MAPS_MUS-', '').replace('-', '_')
            composer = song_name.split('_')[0]
            with open(os.path.join(new_dir, text_path), 'r') as f:
                lines = f.readlines()
                
            last_line = lines[-1].split("\t")
            duration = float(last_line[1])
            split = 'train' if i < train_thresh else 'test'
            year = 2008
            
            new_row = pd.DataFrame({
                'canonical_composer': [composer],
                'canonical_title': [song_name],
                'split': [split],
                'year': [year],
                'midi_filename': [midi_path],
                'audio_filename': [wav_path],
                'duration': [duration]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            i += 1
    return df

os.chdir('/workspace/datasets/maps')
directories = [d for d in os.listdir() if os.path.isdir(d)]
print("DIRECTORIES: ", directories)

df = pd.DataFrame(columns=["canonical_composer", "canonical_title", "split", "year", "midi_filename", "audio_filename", "duration"])

for directory in directories:
    df = split_dir(directory, df)

print(df.head)
print('-'*100)
print(df['split'].value_counts(normalize=True))
print('-'*100)
print(df['duration'].describe())
print('-'*100)
print('exporting to csv...')
output_path = '/workspace/datasets/maestro/maps.csv'
df.to_csv(output_path, index=False)
print('done - {}'.format(output_path))