import wave
import contextlib
import os

def get_wav_duration(file_path):
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

folder = 'dataset/shanghai/WAV'  # 4.321h

total_duration = 0.0

for filename in os.listdir(folder):
    if filename.lower().endswith('.wav'):
        file_path = os.path.join(folder, filename)
        try:
            duration = get_wav_duration(file_path)
            total_duration += duration
        except Exception as e:
            print(f'Error reading {file_path}: {e}')

print(f"Total duration: {total_duration:.2f} seconds")
