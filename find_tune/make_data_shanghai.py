"""
上海方言数据集预处理脚本
从 TXT、TXT_CN 和 WAV 文件生成训练用的 JSONL 文件

级联模式：同时保存上海话文本(text)和普通话文本(text_cn)，
         训练时用上海话，评估时可以计算到普通话的最终 CER
"""
import os
import glob
import json
import re

def parse_txt_file(txt_path):
    """
    解析 TXT 文件，返回 {(start, end, speaker_id, gender): text} 的字典
    """
    result = {}
    if not os.path.exists(txt_path):
        return result
    
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                match = re.match(r'\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s+(\S+)\s+(.+)', line)
                if not match:
                    continue
                
                start, end, speaker_id, gender, text = match.groups()
                start, end = float(start), float(end)
                
                # 过滤掉特殊标记
                if text in ['[ENS]', '[NOISE]']:
                    continue
                
                key = (start, end, speaker_id, gender)
                result[key] = text.strip()
                
            except Exception:
                continue
    
    return result

def process_shanghai_dataset(
    root_dir='dataset/shanghai/',
    wav_dir='WAV',
    txt_dir='TXT',
    txt_cn_dir='TXT_CN',
    output_jsonl='shanghai_hf_data.jsonl',
    use_chinese=False,
    include_both=True
):
    """
    处理上海方言数据集
    
    Args:
        root_dir: 数据集根目录
        wav_dir: WAV 文件目录
        txt_dir: 上海方言文本目录
        txt_cn_dir: 普通话翻译文本目录
        output_jsonl: 输出的 JSONL 文件名
        use_chinese: 是否使用普通话翻译作为主标注（True）还是使用上海方言（False）
        include_both: 是否同时包含上海话和普通话文本（用于级联模式评估）
    """
    all_samples = []
    
    # 获取所有 WAV 文件
    wav_files = sorted(glob.glob(os.path.join(root_dir, wav_dir, '*.wav')))
    
    print(f"Found {len(wav_files)} WAV files")
    
    for wav_path in wav_files:
        wav_file = os.path.basename(wav_path)
        base_name = os.path.splitext(wav_file)[0]
        
        # 读取上海话文本
        txt_sh_path = os.path.join(root_dir, txt_dir, base_name + '.txt')
        # 读取普通话文本
        txt_cn_path = os.path.join(root_dir, txt_cn_dir, base_name + '.txt')
        
        # 解析两个文件
        sh_texts = parse_txt_file(txt_sh_path)
        cn_texts = parse_txt_file(txt_cn_path)
        
        if not sh_texts:
            print(f'Warning: Shanghai TXT file not found or empty for {wav_file}')
            continue
        
        # 遍历上海话文本，匹配普通话翻译
        for key, sh_text in sh_texts.items():
            start, end, speaker_id, gender = key
            
            # 过滤过短的片段
            if (end - start) < 0.5:
                continue
            
            # 获取对应的普通话文本
            cn_text = cn_texts.get(key, "")
            
            sample = {
                "audio_path": wav_path,
                "start": start,
                "end": end,
                "speaker_id": speaker_id,
                "gender": gender,
            }
            
            if include_both:
                # 级联模式：同时保存两种文本
                # text: 训练标签（上海话）
                # text_cn: 评估参考（普通话）
                sample["text"] = sh_text
                sample["text_cn"] = cn_text if cn_text else sh_text  # 如果没有普通话翻译，用上海话代替
            else:
                # 单一模式
                sample["text"] = cn_text if use_chinese else sh_text
            
            all_samples.append(sample)
    
    # 保存为 JSONL
    output_path = os.path.join(root_dir, output_jsonl)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for sample in all_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f'\n✓ Data processing completed!')
    print(f'  Output file: {output_path}')
    print(f'  Total samples: {len(all_samples)}')
    if include_both:
        print(f'  Mode: 级联模式（同时包含上海话和普通话文本）')
        print(f'    - text: 上海话（训练标签）')
        print(f'    - text_cn: 普通话（评估参考）')
    else:
        print(f'  Using {"Chinese (普通话)" if use_chinese else "Shanghai dialect (上海方言)"} as labels')
    
    return output_path, len(all_samples)

if __name__ == "__main__":
    # 级联模式：同时保存上海话和普通话文本
    # text: 上海话（用于训练）
    # text_cn: 普通话（用于评估最终效果）
    process_shanghai_dataset(use_chinese=False, include_both=True)