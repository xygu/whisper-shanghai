"""
加载上海话到普通话翻译数据集
从 TXT（上海话）和 TXT_CN（普通话）文件中提取平行语料
"""
import os
import glob
import json
import re
from datasets import Dataset, DatasetDict
from typing import List, Dict, Tuple, Optional


# 需要清洗的特殊标记
SPECIAL_MARKERS = [
    r'\[\+\]',      # [+] 语气词/未完成标记
    r'\[\*\]',      # [*] 噪音/无效内容
    r'\[\?\]',      # [?] 不确定内容
    r'\[PII\]',     # [PII] 个人身份信息脱敏
    r'\[\]',        # [] 空标记
    r'\[ENS\]',     # [ENS] 结束标记
    r'\[NOISE\]',   # [NOISE] 噪音标记
    r'\[LAUGHTER\]', # [LAUGHTER] 笑声
    r'\[COUGH\]',   # [COUGH] 咳嗽
    r'\[BREATH\]',  # [BREATH] 呼吸声
]


def clean_text(text: str) -> str:
    """
    清洗文本，移除特殊标记
    
    Args:
        text: 原始文本
    
    Returns:
        清洗后的文本
    """
    cleaned = text
    
    # 移除所有特殊标记
    for marker in SPECIAL_MARKERS:
        cleaned = re.sub(marker, '', cleaned)
    
    # 移除其他方括号标记（如 [xxx] 形式的未知标记）
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
    
    # 清理多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 去除首尾空格
    cleaned = cleaned.strip()
    
    return cleaned


def is_valid_text(text: str, min_length: int = 2) -> bool:
    """
    检查文本是否有效
    
    Args:
        text: 文本
        min_length: 最小长度
    
    Returns:
        是否有效
    """
    if not text:
        return False
    
    # 长度检查
    if len(text) < min_length:
        return False
    
    # 检查是否只包含标点符号
    if re.match(r'^[\s\.,，。！？!?、；;：:""\'\'「」【】（）()\[\]]+$', text):
        return False
    
    return True


def parse_txt_line(line: str) -> Optional[Tuple[float, float, str, str, str]]:
    """
    解析 TXT 文件中的一行
    格式: [start,end] speaker_id gender text
    
    Returns:
        (start, end, speaker_id, gender, text) 或 None
    """
    line = line.strip()
    if not line:
        return None
    
    # 使用正则表达式提取时间戳和内容
    match = re.match(r'\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s+(\S+)\s+(.+)', line)
    if not match:
        return None
    
    start, end, speaker_id, gender, text = match.groups()
    return float(start), float(end), speaker_id, gender, text.strip()


def load_parallel_corpus(
    root_dir: str = 'dataset/shanghai/',
    txt_dir: str = 'TXT',
    txt_cn_dir: str = 'TXT_CN',
    clean_data: bool = True,
    min_text_length: int = 2,
) -> List[Dict[str, str]]:
    """
    加载上海话-普通话平行语料
    
    Args:
        root_dir: 数据集根目录
        txt_dir: 上海话文本目录
        txt_cn_dir: 普通话文本目录
        clean_data: 是否清洗数据
        min_text_length: 最小文本长度
    
    Returns:
        平行语料列表，每个元素包含 source（上海话）和 target（普通话）
    """
    parallel_data = []
    skipped_count = 0
    cleaned_count = 0
    
    # 获取所有上海话文本文件
    txt_files = sorted(glob.glob(os.path.join(root_dir, txt_dir, '*.txt')))
    print(f"找到 {len(txt_files)} 个上海话文本文件")
    
    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        txt_cn_path = os.path.join(root_dir, txt_cn_dir, filename)
        
        # 检查对应的普通话文件是否存在
        if not os.path.exists(txt_cn_path):
            print(f"警告: 未找到对应的普通话文件: {filename}")
            continue
        
        # 读取上海话文本
        with open(txt_path, encoding='utf-8') as f:
            shanghai_lines = f.readlines()
        
        # 读取普通话文本
        with open(txt_cn_path, encoding='utf-8') as f:
            mandarin_lines = f.readlines()
        
        # 解析并对齐
        shanghai_dict = {}
        for line in shanghai_lines:
            parsed = parse_txt_line(line)
            if parsed:
                start, end, speaker_id, gender, text = parsed
                key = f"{start:.3f}_{end:.3f}"
                shanghai_dict[key] = text
        
        mandarin_dict = {}
        for line in mandarin_lines:
            parsed = parse_txt_line(line)
            if parsed:
                start, end, speaker_id, gender, text = parsed
                key = f"{start:.3f}_{end:.3f}"
                mandarin_dict[key] = text
        
        # 匹配平行语料
        for key in shanghai_dict:
            if key in mandarin_dict:
                source_text = shanghai_dict[key]
                target_text = mandarin_dict[key]
                
                # 清洗数据
                if clean_data:
                    original_source = source_text
                    original_target = target_text
                    source_text = clean_text(source_text)
                    target_text = clean_text(target_text)
                    
                    if source_text != original_source or target_text != original_target:
                        cleaned_count += 1
                
                # 过滤无效文本
                if not is_valid_text(source_text, min_text_length):
                    skipped_count += 1
                    continue
                if not is_valid_text(target_text, min_text_length):
                    skipped_count += 1
                    continue
                
                # 过滤掉源和目标完全相同的
                if source_text == target_text:
                    skipped_count += 1
                    continue
                
                parallel_data.append({
                    'source': source_text,
                    'target': target_text,
                    'file': filename
                })
    
    print(f"提取到 {len(parallel_data)} 条有效平行语料")
    print(f"清洗了 {cleaned_count} 条数据中的特殊标记")
    print(f"过滤了 {skipped_count} 条无效数据")
    return parallel_data


def create_translation_dataset(
    root_dir: str = 'dataset/shanghai/',
    output_dir: str = 'dataset/shanghai/translation_dataset',
    output_jsonl: str = 'dataset/shanghai/translation_data.jsonl',
    test_size: float = 0.1,
    seed: int = 42,
    clean_data: bool = True,
    min_text_length: int = 2,
) -> DatasetDict:
    """
    创建翻译数据集
    
    Args:
        root_dir: 数据集根目录
        output_dir: 输出目录
        output_jsonl: 输出的 JSONL 文件路径
        test_size: 测试集比例
        seed: 随机种子
        clean_data: 是否清洗数据
        min_text_length: 最小文本长度
    
    Returns:
        DatasetDict 包含 train 和 test 分割
    """
    # 加载平行语料
    parallel_data = load_parallel_corpus(
        root_dir, 
        clean_data=clean_data,
        min_text_length=min_text_length
    )
    
    if len(parallel_data) == 0:
        raise ValueError("未找到有效的平行语料！请检查数据目录。")
    
    # 保存为 JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in parallel_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ 平行语料已保存到: {output_jsonl}")
    
    # 创建 Dataset
    dataset = Dataset.from_list(parallel_data)
    
    # 划分训练集和测试集
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    
    dataset_dict = DatasetDict({
        'train': splits['train'],
        'test': splits['test']
    })
    
    # 保存到磁盘
    dataset_dict.save_to_disk(output_dir)
    print(f"✓ 数据集已保存到: {output_dir}")
    
    # 打印统计信息
    print(f"\n=== 数据集统计 ===")
    print(f"训练集样本数: {len(dataset_dict['train'])}")
    print(f"测试集样本数: {len(dataset_dict['test'])}")
    print(f"\n示例数据:")
    for i in range(min(5, len(dataset_dict['train']))):
        sample = dataset_dict['train'][i]
        print(f"  [{i+1}] 上海话: {sample['source']}")
        print(f"      普通话: {sample['target']}")
        print()
    
    return dataset_dict


def load_translation_dataset(
    dataset_path: str = 'dataset/shanghai/translation_dataset'
) -> DatasetDict:
    """
    加载已保存的翻译数据集
    
    Args:
        dataset_path: 数据集路径
    
    Returns:
        DatasetDict
    """
    from datasets import load_from_disk
    return load_from_disk(dataset_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="创建上海话到普通话翻译数据集")
    parser.add_argument("--root_dir", type=str, default="dataset/shanghai/",
                        help="数据集根目录")
    parser.add_argument("--output_dir", type=str, 
                        default="dataset/shanghai/translation_dataset",
                        help="输出目录")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="测试集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--no_clean", action="store_true",
                        help="不清洗数据")
    parser.add_argument("--min_length", type=int, default=2,
                        help="最小文本长度")
    
    args = parser.parse_args()
    
    create_translation_dataset(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        clean_data=not args.no_clean,
        min_text_length=args.min_length,
    )
