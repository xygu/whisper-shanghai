from datasets import load_dataset, DatasetDict
import numpy as np
import librosa
import os

def load_data(root_dir = "dataset/shanghai/", jsonl_dir = 'shanghai_hf_data.jsonl', out_dir = 'shanghai_dataset', seed = 42):
    out_file = os.path.join(root_dir, out_dir)
    if not os.path.exists(out_file):
        dataset = load_dataset("json", data_files={"full": jsonl_dir})["full"]

        # 2. 随机shuffle
        dataset = dataset.shuffle(seed=seed)

        # 3. 划分train/test（8:2）
        splits = dataset.train_test_split(test_size=0.2, seed=seed)
        train_ds, test_ds = splits["train"], splits["test"]

        # 4. 定义音频切片与重采样（22050→16000）
        def process_audio(item):
            audio_path = item["audio_path"]
            start, end = item["start"], item["end"]
            # 注意 librosa.load 的 path 可以是绝对/相对路径
            arr, _ = librosa.load(audio_path, sr=16000, offset=start, duration=end-start)
            item["audio"] = {"array": arr, "sampling_rate": 16000}
            return item
        ds = DatasetDict()
        # 5. 应用到数据集（推荐dataloader时按需加载时再处理，若数据量不大也可提前map）
        ds['train'] = train_ds.map(process_audio, num_proc=1)
        ds['test']  = test_ds.map(process_audio, num_proc=1)

        ds.save_to_disk(os.path.join(root_dir, out_dir))
        print(f'train and test data saved to {os.path.join(root_dir, out_dir)}')
    else:
        print(f'find {out_file}. loading...')
        from datasets import load_from_disk

        ds = load_from_disk(out_file)   
    print('Train sample:', len(ds['train']))
    print('Test sample:', len(ds['test']))
    print('Dataset keys:', list(ds['train'][0].keys()))
    return ds
if __name__ == "__main__":
    ds = load_data(jsonl_dir = "dataset/shanghai/shanghai_hf_data.jsonl")
    print(ds)
    print('a')
# # 6. 使用
# print("Train sample:", len(train_ds[0]["audio"]["array"]), train_ds[0]["text"])
# print("Test  sample:", len(test_ds[0]["audio"]["array"]), test_ds[0]["text"])

# 可直接交给ASR/Whisper等模型用于训练
