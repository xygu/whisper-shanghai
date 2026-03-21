"""
统一数据加载脚本
确保所有训练方式（full、lora、e2e、joint）使用完全相同的 train/test 划分

数据集包含以下字段：
- audio: 音频数据
- text: 上海话文本（用于级联模式 ASR 训练）
- text_cn: 普通话文本（用于端到端模式训练和最终评估）
"""
from datasets import load_dataset, DatasetDict, Audio, load_from_disk
import numpy as np
import librosa
import os
import multiprocessing

# ==================== 统一配置 ====================
UNIFIED_CONFIG = {
    "root_dir": "dataset/shanghai/",
    "jsonl_file": "shanghai_unified_data.jsonl",  # 统一的 JSONL 文件
    "dataset_dir": "shanghai_unified_dataset",     # 统一的数据集目录
    "seed": 42,                                    # 固定随机种子
    "test_size": 0.2,                              # 测试集比例 20%
}


def get_optimal_num_proc() -> int:
    """返回预处理进程数（固定为1，单线程更稳定）"""
    return 1


def load_unified_dataset(
    root_dir: str = None,
    jsonl_file: str = None,
    dataset_dir: str = None,
    seed: int = None,
    test_size: float = None,
    force_rebuild: bool = False,
):
    """
    加载统一的上海方言数据集
    
    所有训练方式（full、lora、e2e、joint）都应该调用此函数，
    确保使用完全相同的 train/test 划分。
    
    Args:
        root_dir: 数据集根目录（默认使用统一配置）
        jsonl_file: JSONL 文件名（默认使用统一配置）
        dataset_dir: 数据集目录名（默认使用统一配置）
        seed: 随机种子（默认使用统一配置）
        test_size: 测试集比例（默认使用统一配置）
        force_rebuild: 是否强制重建数据集
    
    Returns:
        DatasetDict: 包含 train 和 test 的数据集
    """
    # 使用统一配置
    root_dir = root_dir or UNIFIED_CONFIG["root_dir"]
    jsonl_file = jsonl_file or UNIFIED_CONFIG["jsonl_file"]
    dataset_dir = dataset_dir or UNIFIED_CONFIG["dataset_dir"]
    seed = seed if seed is not None else UNIFIED_CONFIG["seed"]
    test_size = test_size if test_size is not None else UNIFIED_CONFIG["test_size"]
    
    dataset_path = os.path.join(root_dir, dataset_dir)
    jsonl_path = os.path.join(root_dir, jsonl_file)
    
    # 检查是否已存在处理好的数据集
    if os.path.exists(dataset_path) and not force_rebuild:
        print(f"✓ 加载已有数据集: {dataset_path}")
        ds = load_from_disk(dataset_path)
        _print_dataset_summary(ds)
        return ds
    
    # 检查 JSONL 文件是否存在
    if not os.path.exists(jsonl_path):
        print(f"❌ JSONL 文件不存在: {jsonl_path}")
        print("请先运行数据准备脚本:")
        print("  python find_tune/make_data_unified.py")
        return None
    
    print(f"创建统一数据集...")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Seed: {seed}")
    print(f"  Test size: {test_size}")
    
    # 1. 加载 JSONL 文件
    dataset = load_dataset("json", data_files={"full": jsonl_path})["full"]
    print(f"  总样本数: {len(dataset)}")
    
    # 2. 随机 shuffle（使用固定种子）
    dataset = dataset.shuffle(seed=seed)
    
    # 3. 划分 train/test（使用固定种子）
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    train_ds, test_ds = splits["train"], splits["test"]
    
    print(f"  Train: {len(train_ds)}")
    print(f"  Test: {len(test_ds)}")
    
    # 4. 处理音频
    def process_audio(item):
        """处理音频：加载、切片、重采样到 16kHz"""
        audio_path = item["audio_path"]
        start, end = item["start"], item["end"]
        
        try:
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
            print(f"Error processing audio {audio_path} [{start}, {end}]: {e}")
            item["audio"] = {
                "array": np.zeros(1600, dtype=np.float32),
                "sampling_rate": 16000
            }
        
        return item
    
    # 5. 应用音频处理
    num_proc = get_optimal_num_proc()
    print(f"处理音频（{num_proc} 进程）...")
    
    ds = DatasetDict()
    ds["train"] = train_ds.map(process_audio, num_proc=num_proc)
    ds["test"] = test_ds.map(process_audio, num_proc=num_proc)
    
    # 6. 保存到磁盘
    ds.save_to_disk(dataset_path)
    print(f"✓ 数据集已保存: {dataset_path}")
    
    _print_dataset_summary(ds)
    return ds


def _print_dataset_summary(ds: DatasetDict):
    """打印数据集摘要"""
    print(f"\n{'='*50}")
    print(f"统一数据集摘要")
    print(f"{'='*50}")
    print(f"Train 样本数: {len(ds['train'])}")
    print(f"Test 样本数: {len(ds['test'])}")
    print(f"字段: {list(ds['train'][0].keys())}")
    
    sample = ds["train"][0]
    print(f"\n示例:")
    print(f"  上海话 (text): {sample.get('text', 'N/A')}")
    print(f"  普通话 (text_cn): {sample.get('text_cn', 'N/A')}")
    print(f"  音频长度: {len(sample['audio']['array'])} samples")
    print(f"{'='*50}\n")


def get_dataset_for_mode(mode: str = "cascade"):
    """
    根据训练模式获取数据集
    
    Args:
        mode: 训练模式
            - "cascade": 级联模式（ASR: 上海话语音 -> 上海话文本）
            - "e2e": 端到端模式（上海话语音 -> 普通话文本）
            - "joint": 联合训练模式（同时训练 ASR 和翻译）
    
    Returns:
        DatasetDict: 数据集
    """
    ds = load_unified_dataset()
    
    if ds is None:
        return None
    
    print(f"训练模式: {mode}")
    
    if mode == "cascade":
        print("  目标: 上海话语音 -> 上海话文本 (text)")
        print("  评估: 使用 text_cn 计算最终 CER")
    elif mode == "e2e":
        print("  目标: 上海话语音 -> 普通话文本 (text_cn)")
    elif mode == "joint":
        print("  目标: 同时训练 ASR (text) 和翻译 (text -> text_cn)")
    
    return ds


# ==================== 兼容旧接口 ====================
def load_shanghai_data(**kwargs):
    """兼容旧的 load_data_shanghai.py 接口"""
    print("⚠️ 警告: load_shanghai_data() 已废弃，请使用 load_unified_dataset()")
    return load_unified_dataset(**kwargs)


def load_shanghai_e2e_data(**kwargs):
    """兼容旧的 load_data_shanghai_e2e.py 接口"""
    print("⚠️ 警告: load_shanghai_e2e_data() 已废弃，请使用 load_unified_dataset()")
    return load_unified_dataset(**kwargs)


if __name__ == "__main__":
    # 测试加载
    ds = load_unified_dataset()
    if ds:
        print("\n✓ 统一数据集加载成功!")
        print(f"  Train: {len(ds['train'])} 样本")
        print(f"  Test: {len(ds['test'])} 样本")
