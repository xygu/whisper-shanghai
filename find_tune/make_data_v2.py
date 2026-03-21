import os
import json

# 设置文件路径
WAV_DIR = "dataset/shanghai/WAV/"
TXT_DIR = "dataset/shanghai/TXT/"

OUTPUT_JSONL = "dataset/shanghai/shanghai_hf_data.jsonl"  # 输出文件

def process_transcripts_and_audio(wav_dir, txt_dir, output_jsonl):
    # 文件路径校验
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
    txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])

    # 校验文件是否匹配
    if len(wav_files) != len(txt_files):
        raise ValueError("音频文件和文本文件数量不匹配，请检查！")
    
    dataset = []

    # 遍历音频文件
    for wav_file, txt_file in zip(wav_files, txt_files):
        wav_path = os.path.join(wav_dir, wav_file)
        txt_path = os.path.join(txt_dir, txt_file)
        transcript_entries = []

        # 解析文本文件
        with open(txt_path, "r", encoding="utf-8") as txt_f:
            lines = txt_f.readlines()

        for line in lines:
            # 解析每一行文本: [start_time, end_time] speaker_id gender text
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            time_range, speaker_id, gender, text = parts
            start_time, end_time = map(float, time_range.strip("[]").split(","))
            
            # 构造单个条目
            transcript_entries.append({
                "start": start_time,
                "end": end_time,
                "speaker_id": speaker_id,
                "gender": gender,
                "text": text,
                "audio_path": wav_path,  # 对应音频文件路径
            })

        # 将结果添加到数据集
        dataset.extend(transcript_entries)

    # 将所有条目写入 JSON Lines 文件
    with open(output_jsonl, "w", encoding="utf-8") as jsonl_f:
        for entry in dataset:
            json.dump(entry, jsonl_f, ensure_ascii=False)
            jsonl_f.write("\n")

    print(f"JSON Lines 文件已保存至：{output_jsonl}")

if __name__ == "__main__":
    process_transcripts_and_audio(WAV_DIR, TXT_DIR, OUTPUT_JSONL)
