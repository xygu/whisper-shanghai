"""
计算级联模式的中文 WER
流程：
1. 使用 Whisper ASR 模型将上海话语音转为上海话文本
2. 使用 MT5 翻译模型将上海话文本翻译为中文
3. 计算中文 WER
"""
import os
import argparse
import json
import torch
from tqdm import tqdm
from datetime import datetime

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import (
    pipeline,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
)
from datasets import load_from_disk, Audio
import evaluate


def compute_wer(predictions, references):
    """计算 WER（词错误率）- 对于中文按字符计算"""
    metric = evaluate.load("wer")
    pred_chars = [" ".join(list(s.replace(" ", ""))) for s in predictions]
    ref_chars = [" ".join(list(s.replace(" ", ""))) for s in references]
    return 100 * metric.compute(predictions=pred_chars, references=ref_chars)


def translate_text(text, tokenizer, model, device, max_length=128):
    """翻译单条上海话到普通话"""
    inputs = tokenizer(
        f"translate Shanghai to Mandarin: {text}",
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="计算级联模式的中文 WER")
    parser.add_argument("--whisper_model", type=str, required=True, help="Whisper ASR 模型路径")
    parser.add_argument("--translation_model", type=str, required=True, help="MT5 翻译模型路径")
    parser.add_argument("--dataset_path", type=str, 
                        default="/mnt/workspace/workgroup/qq/ts/whisper/dataset/shanghai/shanghai_unified_dataset",
                        help="数据集路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认为 whisper_model 目录）")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or args.whisper_model
    
    print("=" * 70)
    print("级联模式中文 WER 评估")
    print("=" * 70)
    print(f"Whisper 模型: {args.whisper_model}")
    print(f"翻译模型: {args.translation_model}")
    print(f"数据集: {args.dataset_path}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = load_from_disk(args.dataset_path)
    test_dataset = dataset["test"].cast_column("audio", Audio(sampling_rate=16000))
    print(f"测试样本数: {len(test_dataset)}")
    
    if "text_cn" not in test_dataset.column_names:
        print("❌ 数据集中没有 text_cn 字段，无法计算中文 WER")
        return
    
    # 加载 Whisper ASR 模型
    print(f"\n加载 Whisper 模型: {args.whisper_model}")
    asr_pipe = pipeline("automatic-speech-recognition", model=args.whisper_model, device=device)
    if hasattr(asr_pipe.model.config, 'forced_decoder_ids'):
        asr_pipe.model.config.forced_decoder_ids = None
    if hasattr(asr_pipe.model.generation_config, 'forced_decoder_ids'):
        asr_pipe.model.generation_config.forced_decoder_ids = None
    print("✓ Whisper 模型加载完成")
    
    # 加载翻译模型
    translation_model_path = args.translation_model
    if not os.path.exists(os.path.join(translation_model_path, "spiece.model")):
        final_model_path = os.path.join(translation_model_path, "final_model")
        if os.path.exists(os.path.join(final_model_path, "spiece.model")):
            translation_model_path = final_model_path
    
    print(f"\n加载翻译模型: {translation_model_path}")
    trans_tokenizer = MT5Tokenizer.from_pretrained(translation_model_path)
    trans_model = MT5ForConditionalGeneration.from_pretrained(translation_model_path).to(device)
    trans_model.eval()
    print("✓ 翻译模型加载完成")
    
    # 开始评估
    print("\n开始评估...")
    
    shanghai_preds = []
    shanghai_refs = []
    mandarin_preds = []
    mandarin_refs = []
    
    for i in tqdm(range(len(test_dataset)), desc="ASR 推理"):
        sample = test_dataset[i]
        audio = sample["audio"]
        
        # ASR 推理
        audio_input = {"array": audio["array"], "sampling_rate": audio["sampling_rate"]}
        result = asr_pipe(audio_input, generate_kwargs={"language": "Chinese", "task": "transcribe"})
        
        shanghai_pred = result['text']
        shanghai_preds.append(shanghai_pred)
        shanghai_refs.append(sample["text"])
        mandarin_refs.append(sample["text_cn"])
    
    # 翻译
    print("\n翻译上海话到普通话...")
    for shanghai_text in tqdm(shanghai_preds, desc="翻译"):
        translated = translate_text(shanghai_text, trans_tokenizer, trans_model, device)
        mandarin_preds.append(translated)
    
    # 计算指标
    print("\n计算指标...")
    shanghai_wer = compute_wer(shanghai_preds, shanghai_refs)
    mandarin_wer = compute_wer(mandarin_preds, mandarin_refs)
    
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"上海话 WER (ASR): {shanghai_wer:.2f}%")
    print(f"中文 WER (级联): {mandarin_wer:.2f}%")
    print("=" * 70)
    
    # 保存结果
    results = {
        "summary": {
            "whisper_model": args.whisper_model,
            "translation_model": args.translation_model,
            "test_samples": len(test_dataset),
            "shanghai_wer": shanghai_wer,
            "mandarin_wer": mandarin_wer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "details": [
            {
                "index": i,
                "shanghai_ref": shanghai_refs[i],
                "shanghai_pred": shanghai_preds[i],
                "mandarin_ref": mandarin_refs[i],
                "mandarin_pred": mandarin_preds[i],
            }
            for i in range(len(test_dataset))
        ]
    }
    
    output_file = os.path.join(output_dir, "cascade_wer_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    # 打印样例
    print("\n样例对比（前5个）")
    print("=" * 70)
    for i in range(min(5, len(test_dataset))):
        print(f"\n样例 {i+1}:")
        print(f"  上海话参考: {shanghai_refs[i]}")
        print(f"  上海话预测: {shanghai_preds[i]}")
        print(f"  中文参考:   {mandarin_refs[i]}")
        print(f"  中文预测:   {mandarin_preds[i]}")


if __name__ == "__main__":
    main()
