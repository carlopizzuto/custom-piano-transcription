import os
import csv
import soundfile as sf

input_csv = './datasets/data/maestro-v2.0.0.csv'
output_csv = './datasets/data/maestro-v2.0.0.1.csv'

with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read header and add 'duration' column
    header = next(reader)
    header.append('duration')
    writer.writerow(header)

    for row in reader:
        audio_path = row[5]  # Adjust index if 'audio_filename' is in a different column
        if not os.path.isabs(audio_path):
            audio_path = os.path.join('./datasets/data', audio_path)
        try:
            f = sf.SoundFile(audio_path)
            duration = len(f) / f.samplerate
            f.close()
        except Exception as e:
            print(f"Error reading {audio_path}: {e}")
            duration = 0.0  # Set duration to 0.0 if there's an error

        row.append(str(duration))
        writer.writerow(row)