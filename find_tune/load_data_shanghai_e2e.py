"""
加载上海方言端到端数据集并创建 HuggingFace Dataset
上海话语音 -> 普通话文本
自动检测硬件配置，智能调整并行度
"""
from datasets import load_dataset, DatasetDict, Audio
import numpy as np
import librosa
import os
import multiprocessing


def get_optimal_num_proc() -> int:
    """
    根据 CPU 核心数自动计算最优的并行进程数
    
    Returns:
        int: 推荐的进程数
    """
    cpu_count = multiprocessing.cpu_count()
    # 使用 CPU 核心数的 75%，避免过载，最少 1 个，最多 16 个
    num_proc = max(1, min(cpu_count * 3 // 4, 16))
    return num_proc

def load_shanghai_e2e_data(
    root_dir="dataset/shanghai/",
    jsonl_file='shanghai_e2e_data.jsonl',
    out_dir='shanghai_e2e_dataset',
    seed=42,
    test_size=0.2
):
    """
    加载上海方言端到端数据集（上海话语音 -> 普通话文本）
    
    Args:
        root_dir: 数据集根目录
        jsonl_file: JSONL 文件名
        out_dir: 输出目录名
        seed: 随机种子
        test_size: 测试集比例
    """
    out_path = os.path.join(root_dir, out_dir)
    jsonl_path = os.path.join(root_dir, jsonl_file)
    
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}")
        print("Please run make_data_shanghai_e2e.py first:")
        print("  python find_tune/make_data_shanghai_e2e.py")
        return None
    
    if os.path.exists(out_path):
        print(f'Dataset already exists at {out_path}. Loading...')
        from datasets import load_from_disk
        ds = load_from_disk(out_path)
    else:
        print(f'Creating end-to-end dataset from {jsonl_path}...')
        print(f'Task: Shanghai Speech -> Mandarin Text')
        
        # 1. 加载 JSONL 文件
        dataset = load_dataset("json", data_files={"full": jsonl_path})["full"]
        print(f'Loaded {len(dataset)} samples')
        
        # 2. 随机 shuffle
        dataset = dataset.shuffle(seed=seed)
        
        # 3. 划分 train/test
        splits = dataset.train_test_split(test_size=test_size, seed=seed)
        train_ds, test_ds = splits["train"], splits["test"]
        
        # 4. 定义音频切片与重采样函数
        def process_audio(item):
            """处理音频：加载、切片、重采样到 16kHz"""
            audio_path = item["audio_path"]
            start, end = item["start"], item["end"]
            
            try:
                # 使用 librosa 加载音频片段
                arr, _ = librosa.load(
                    audio_path,
                    sr=16000,
                    offset=start,
                    duration=end - start
                )
                item["audio"] = {
                    "array": arr,
                    "sampling_rate": 16000
                }
            except Exception as e:
                print(f'Error processing audio {audio_path} [{start}, {end}]: {e}')
                # 如果出错，创建空音频
                item["audio"] = {
                    "array": np.zeros(1600, dtype=np.float32),  # 0.1秒的静音
                    "sampling_rate": 16000
                }
            
            return item
        
        # 5. 应用音频处理（自动检测最优进程数）
        num_proc = get_optimal_num_proc()
        print(f'Processing audio files with {num_proc} processes (auto-detected from {multiprocessing.cpu_count()} CPU cores)...')
        ds = DatasetDict()
        ds['train'] = train_ds.map(process_audio, num_proc=num_proc)
        ds['test'] = test_ds.map(process_audio, num_proc=num_proc)
        
        # 6. 保存到磁盘
        ds.save_to_disk(out_path)
        print(f'✓ Dataset saved to {out_path}')
    
    print(f'\n=== End-to-End Dataset Summary ===')
    print(f'Task: Shanghai Speech -> Mandarin Text')
    print(f'Train samples: {len(ds["train"])}')
    print(f'Test samples: {len(ds["test"])}')
    print(f'Dataset keys: {list(ds["train"][0].keys())}')
    print(f'Target text (普通话): {ds["train"][0]["text"]}')
    if "source_text" in ds["train"][0]:
        print(f'Source text (上海话): {ds["train"][0]["source_text"]}')
    print(f'Audio length: {len(ds["train"][0]["audio"]["array"])} samples')
    
    return ds

if __name__ == "__main__":
    ds = load_shanghai_e2e_data()
    if ds:
        print("\n✓ End-to-end data loading completed successfully!")
