"""
统一数据准备脚本
生成包含所有字段的 JSONL 文件，供所有训练方式共用

字段说明：
- audio_path: 音频文件路径
- start: 开始时间
- end: 结束时间
- speaker_id: 说话人 ID
- gender: 性别
- text: 上海话文本（用于级联模式 ASR 训练）
- text_cn: 普通话文本（用于端到端模式和最终评估）
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


def make_unified_dataset(
    root_dir='dataset/shanghai/',
    wav_dir='WAV',
    txt_dir='TXT',
    txt_cn_dir='TXT_CN',
    output_jsonl='shanghai_unified_data.jsonl',
    min_duration=0.5,
):
    """
    生成统一的 JSONL 数据文件
    
    只保留同时有上海话和普通话文本的样本，确保所有训练方式使用相同的数据。
    
    Args:
        root_dir: 数据集根目录
        wav_dir: WAV 文件目录
        txt_dir: 上海方言文本目录
        txt_cn_dir: 普通话翻译文本目录
        output_jsonl: 输出的 JSONL 文件名
        min_duration: 最小音频时长（秒）
    """
    all_samples = []
    skipped_no_cn = 0
    skipped_short = 0
    
    # 获取所有 WAV 文件
    wav_files = sorted(glob.glob(os.path.join(root_dir, wav_dir, '*.wav')))
    
    print(f"找到 {len(wav_files)} 个 WAV 文件")
    print(f"生成统一数据集...")
    
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
            continue
        
        # 遍历上海话文本，匹配普通话翻译
        for key, sh_text in sh_texts.items():
            start, end, speaker_id, gender = key
            
            # 过滤过短的片段
            if (end - start) < min_duration:
                skipped_short += 1
                continue
            
            # 获取对应的普通话文本
            cn_text = cn_texts.get(key, "")
            
            # 只保留同时有上海话和普通话文本的样本
            if not cn_text:
                skipped_no_cn += 1
                continue
            
            sample = {
                "audio_path": wav_path,
                "start": start,
                "end": end,
                "speaker_id": speaker_id,
                "gender": gender,
                "text": sh_text,      # 上海话文本
                "text_cn": cn_text,   # 普通话文本
            }
            
            all_samples.append(sample)
    
    # 保存为 JSONL
    output_path = os.path.join(root_dir, output_jsonl)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for sample in all_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*50}")
    print(f"统一数据集生成完成!")
    print(f"{'='*50}")
    print(f"输出文件: {output_path}")
    print(f"总样本数: {len(all_samples)}")
    print(f"跳过（无普通话翻译）: {skipped_no_cn}")
    print(f"跳过（时长过短）: {skipped_short}")
    print(f"\n字段说明:")
    print(f"  - text: 上海话文本（级联模式 ASR 训练）")
    print(f"  - text_cn: 普通话文本（端到端模式 / 最终评估）")
    
    # 打印示例
    if all_samples:
        print(f"\n示例:")
        print(f"  上海话: {all_samples[0]['text']}")
        print(f"  普通话: {all_samples[0]['text_cn']}")
    
    print(f"\n下一步:")
    print(f"  运行 python find_tune/load_data_unified.py 创建数据集")
    
    return output_path, len(all_samples)


if __name__ == "__main__":
    make_unified_dataset()
