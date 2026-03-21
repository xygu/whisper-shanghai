"""
上海方言端到端数据集预处理脚本
从上海话语音直接到普通话文本（跳过上海话文本）
"""
import os
import glob
import json
import re

def process_shanghai_e2e_dataset(
    root_dir='dataset/shanghai/',
    wav_dir='WAV',
    txt_dir='TXT',
    txt_cn_dir='TXT_CN',
    output_jsonl='shanghai_e2e_data.jsonl'
):
    """
    处理上海方言端到端数据集（上海话语音 -> 普通话文本）
    
    Args:
        root_dir: 数据集根目录
        wav_dir: WAV 文件目录
        txt_dir: 上海方言文本目录（用于对齐时间戳）
        txt_cn_dir: 普通话翻译文本目录（作为目标标签）
        output_jsonl: 输出的 JSONL 文件名
    """
    all_samples = []
    
    # 获取所有 WAV 文件
    wav_files = sorted(glob.glob(os.path.join(root_dir, wav_dir, '*.wav')))
    
    print(f"Found {len(wav_files)} WAV files")
    print(f"Building end-to-end dataset: Shanghai Speech -> Mandarin Text")
    
    for wav_path in wav_files:
        wav_file = os.path.basename(wav_path)
        base_name = os.path.splitext(wav_file)[0]
        
        # 上海方言文本路径（用于获取时间戳和对齐信息）
        txt_sh_path = os.path.join(root_dir, txt_dir, base_name + '.txt')
        # 普通话翻译文本路径（作为目标标签）
        txt_cn_path = os.path.join(root_dir, txt_cn_dir, base_name + '.txt')
        
        if not os.path.exists(txt_sh_path):
            print(f'Warning: Shanghai TXT file not found for {wav_file}')
            continue
        if not os.path.exists(txt_cn_path):
            print(f'Warning: Mandarin TXT file not found for {wav_file}')
            continue
        
        # 读取上海方言文本（获取时间戳）
        sh_lines = []
        with open(txt_sh_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sh_lines.append(line)
        
        # 读取普通话翻译文本
        cn_lines = []
        with open(txt_cn_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cn_lines.append(line)
        
        # 对齐上海话和普通话文本
        for sh_line, cn_line in zip(sh_lines, cn_lines):
            try:
                # 解析上海话格式: [start,end] speaker_id gender text
                sh_match = re.match(r'\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s+(\S+)\s+(.+)', sh_line)
                if not sh_match:
                    continue
                
                start, end, speaker_id, gender, sh_text = sh_match.groups()
                start, end = float(start), float(end)
                
                # 解析普通话格式: [start,end] speaker_id gender text
                cn_match = re.match(r'\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s+(\S+)\s+(.+)', cn_line)
                if not cn_match:
                    continue
                
                cn_text = cn_match.groups()[4]
                
                # 过滤掉特殊标记和过短的片段
                if sh_text in ['[ENS]', '[NOISE]'] or cn_text in ['[ENS]', '[NOISE]']:
                    continue
                if (end - start) < 0.5:
                    continue
                
                sample = {
                    "audio_path": wav_path,
                    "start": start,
                    "end": end,
                    "speaker_id": speaker_id,
                    "gender": gender,
                    "text": cn_text.strip(),  # 使用普通话文本作为目标
                    "source_text": sh_text.strip()  # 保留上海话原文用于参考
                }
                all_samples.append(sample)
                
            except Exception as e:
                print(f'Error parsing line: {sh_line}, Error: {e}')
                continue
    
    # 保存为 JSONL
    output_path = os.path.join(root_dir, output_jsonl)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for sample in all_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f'\n✓ End-to-end data processing completed!')
    print(f'  Output file: {output_path}')
    print(f'  Total samples: {len(all_samples)}')
    print(f'  Task: Shanghai Speech -> Mandarin Text (端到端)')
    
    # 打印示例
    if all_samples:
        print(f'\n=== Sample ===')
        print(f'  Source (上海话): {all_samples[0]["source_text"]}')
        print(f'  Target (普通话): {all_samples[0]["text"]}')
    
    return output_path, len(all_samples)

if __name__ == "__main__":
    process_shanghai_e2e_dataset()
