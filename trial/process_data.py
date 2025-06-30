import os
import re
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

# --- 1. 配置你的文件路径 ---
# 源文件路径
root_dir = 'dataset/shanghai/'
WAV_DIR = os.path.join(root_dir, "WAV")
TXT_CN_DIR = os.path.join(root_dir, "TXT_CN")

# 输出路径
OUTPUT_AUDIO_DIR = os.path.join(root_dir, "segmented_audio") # 存放切分后音频的文件夹
OUTPUT_METADATA_FILE = os.path.join(root_dir, "metadata.csv") # 最终生成的清单文件

# --- 2. 创建输出文件夹 ---
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

# --- 3. 主处理逻辑 ---
def process_data():
    """
    遍历所有翻译文件，切割对应的音频，并生成 metadata.csv
    """
    all_records = []
    
    # 编译正则表达式以提高效率，用于解析 [11.660,14.350] G0004 female ... 格式的行
    # 这个正则表达式会捕获：开始时间, 结束时间, 说话人ID, 性别, 以及后面的所有文本
    line_pattern = re.compile(r'\[(\d+\.\d+),(\d+\.\d+)\]\s+([^\s]+)\s+([^\s]+)\s+(.*)')

    # 获取所有待处理的翻译文件名
    translation_files = [f for f in os.listdir(TXT_CN_DIR) if f.endswith('.txt')]
    
    print(f"发现 {len(translation_files)} 个翻译文件，开始处理...")

    # 使用 tqdm 创建进度条
    for txt_filename in tqdm(translation_files, desc="处理对话"):
        # 从翻译文件名推导出音频文件名
        # 例如：A0002_S0003_0_G0003_G0004_cv.txt -> A0002_S0003_0_G0003_G0004
        base_name = txt_filename.replace('_cv.txt', '')
        wav_filename = f"{base_name}.wav"
        wav_filepath = os.path.join(WAV_DIR, wav_filename)
        
        txt_filepath = os.path.join(TXT_CN_DIR, txt_filename)

        # 检查对应的音频文件是否存在
        if not os.path.exists(wav_filepath):
            print(f"警告：找不到对应的音频文件 {wav_filepath}，跳过 {txt_filename}")
            continue

        try:
            # 加载完整的对话音频
            conversation_audio = AudioSegment.from_wav(wav_filepath)
            
            # 读取并解析翻译文件
            with open(txt_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    match = line_pattern.match(line.strip())
                    if not match:
                        continue # 如果某一行格式不匹配，则跳过

                    # 从正则匹配中提取信息
                    start_time_s, end_time_s, speaker_id, gender, mandarin_text = match.groups()
                    
                    # pydub 使用毫秒作为单位，需要转换
                    start_ms = float(start_time_s) * 1000
                    end_ms = float(end_time_s) * 1000

                    # 切割音频
                    audio_segment = conversation_audio[start_ms:end_ms]

                    # 定义切割后音频的输出路径和文件名
                    # 文件名格式：基础名_说话人ID_开始时间ms_结束时间ms.wav
                    segment_filename = f"{base_name}_{speaker_id}_{int(start_ms)}_{int(end_ms)}.wav"
                    segment_filepath = os.path.join(OUTPUT_AUDIO_DIR, segment_filename)

                    # 导出切割后的音频
                    audio_segment.export(segment_filepath, format="wav")

                    # 将记录添加到总列表中
                    all_records.append({
                        "audio_path": segment_filepath,
                        "transcription": mandarin_text.strip(),
                        "speaker_id": speaker_id,
                        "duration_ms": end_ms - start_ms
                    })
        
        except Exception as e:
            print(f"处理文件 {txt_filename} 时发生错误: {e}")

    # --- 4. 创建并保存 DataFrame ---
    if not all_records:
        print("错误：没有处理任何数据，请检查文件路径和格式。")
        return

    df = pd.DataFrame(all_records)
    
    # 剔除过短或过长的音频（可选，但推荐）
    min_duration = 500  # 0.5秒
    max_duration = 20000 # 20秒
    df_filtered = df[(df['duration_ms'] >= min_duration) & (df['duration_ms'] <= max_duration)]

    # 保存到 CSV 文件
    df_filtered.to_csv(OUTPUT_METADATA_FILE, index=False, encoding='utf-8')
    
    print("\n处理完成！")
    print(f"总共生成了 {len(df_filtered)} 条可用于训练的数据。")
    print(f"清单文件已保存到: {OUTPUT_METADATA_FILE}")
    print("\n清单文件内容预览:")
    print(df_filtered.head())


if __name__ == "__main__":
    process_data()