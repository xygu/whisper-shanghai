import os
import glob
import json
import re
# 20251215采用这个版本
root_dir = 'dataset/shanghai/'
wav_dir = 'WAV'
txt_dir = 'TXT'
output_jsonl = 'shanghai_hf_data.jsonl'

all_samples = []

wav_files = sorted(glob.glob(os.path.join(root_dir,wav_dir, '*.wav')))

for wav_path in wav_files:
    wav_file = os.path.basename(wav_path)
    base_name = os.path.splitext(wav_file)[0]
    txt_path = os.path.join(root_dir, txt_dir, base_name + '.txt')
    
    if not os.path.exists(txt_path):
        print(f'Warning: TXT file not found for {wav_file}')
        continue
        
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            time_str, speaker_id, gender, text = re.split(r'\s+', line.strip(),3)
            time_str = time_str.strip('[]')
            start, end = map(float, time_str.split(','))
            sample = {
                "audio_path": wav_path,
                "start": start,
                "end": end,
                "speaker_id": speaker_id,
                "gender": gender,
                "text": text
            }
            all_samples.append(sample)

# 保存为jsonl
with open(os.path.join(root_dir,output_jsonl), 'w', encoding='utf-8') as fout:
    for sample in all_samples:
        fout.write(json.dumps(sample, ensure_ascii=False)+'\n')

print('Done:', output_jsonl, 'Total samples:', len(all_samples))
