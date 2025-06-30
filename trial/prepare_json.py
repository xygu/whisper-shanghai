import os, json
from tqdm import tqdm

# 输入路径
root_dir = 'dataset/shanghai'
input_txt_path = os.path.join(root_dir, "TXT")
input_audio_path = os.path.join(root_dir, "WAV")
output_jsonl = os.path.join(root_dir, "whisper_finetune_data.jsonl")

all_data = []

for filename in os.listdir(input_txt_path):
    if not filename.endswith(".txt"): continue
    basename = filename[:-4]
    audio_file = os.path.join(input_audio_path, f"{basename}.wav")
    text_file = os.path.join(input_txt_path, filename)

    with open(text_file, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip() == "": continue
            if not line.startswith("["): continue
            time_info, rest = line.split("]", 1)
            start, end = map(float, time_info[1:].split(","))
            text = rest.split(maxsplit=2)[-1].strip()
            item = {
                "audio": audio_file,
                "start": start,
                "end": end,
                "text": text
            }
            all_data.append(item)

with open(output_jsonl, 'w', encoding='utf8') as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"共提取 {len(all_data)} 条数据，保存在 {output_jsonl}")