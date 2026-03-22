#!/usr/bin/env python3
"""
ASR-TACT: Targeted Activation Concept Tuning for ASR

主运行脚本，支持以下功能：
1. train_sae: 训练 SAE 模型
2. analyze: 分析神经元激活和特征关联
3. train_gated_lora: 训练门控 LoRA 模型
4. inference: 使用训练好的模型进行推理

使用示例:
    # 训练 SAE
    python run_asr_tact.py train_sae --config asr_tact/config.yaml
    
    # 分析神经元
    python run_asr_tact.py analyze --sae_checkpoint exp/asr_tact/sae/sae-layer12-260322-042557/best_model.pt
    
    # 训练门控 LoRA
    python run_asr_tact.py train_gated_lora --sae_checkpoint exp/sae/best_model.pt --analysis_dir exp/neuron_analysis
"""

import os
import sys
import argparse
import yaml
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def cmd_train_sae(args):
    """训练 SAE 模型"""
    from asr_tact.train_sae import train_sae
    
    config = load_config(args.config) if args.config else {}
    sae_config = config.get('sae', {})
    training_config = config.get('sae_training', {})
    
    train_sae(
        model_name=args.model_name or config.get('whisper', {}).get('model_name', 'openai/whisper-medium'),
        dataset_path=args.dataset_path or config.get('dataset', {}).get('path', 'dataset/shanghai/shanghai_dataset'),
        output_dir=args.output_dir or os.path.join(config.get('output', {}).get('base_dir', './exp/asr_tact'), 'sae'),
        encoder_layer=args.encoder_layer or sae_config.get('encoder_layer', 12),
        latent_dim=args.latent_dim or sae_config.get('latent_dim', 8192),
        topk=args.topk or sae_config.get('topk', 64),
        norm_type=args.norm_type or sae_config.get('norm_type', 'z-norm'),
        batch_size=args.batch_size or training_config.get('batch_size', 4),
        num_epochs=args.num_epochs or training_config.get('num_epochs', 10),
        learning_rate=args.learning_rate or training_config.get('learning_rate', 1e-4),
        device=args.device or config.get('hardware', {}).get('device', 'cuda'),
        save_every=args.save_every or training_config.get('save_every', 1000),
        log_every=args.log_every or training_config.get('log_every', 100),
    )


def cmd_analyze(args):
    """分析神经元"""
    from asr_tact.analyze_neurons import analyze_neurons
    
    config = load_config(args.config) if args.config else {}
    analysis_config = config.get('neuron_analysis', {})
    
    analyze_neurons(
        sae_checkpoint=args.sae_checkpoint,
        model_name=args.model_name or config.get('whisper', {}).get('model_name', 'openai/whisper-medium'),
        dataset_path=args.dataset_path or config.get('dataset', {}).get('path', 'dataset/shanghai/shanghai_dataset'),
        output_dir=args.output_dir or os.path.join(config.get('output', {}).get('base_dir', './exp/asr_tact'), 'neuron_analysis'),
        encoder_layer=args.encoder_layer or config.get('sae', {}).get('encoder_layer', 12),
        topk=args.topk or config.get('sae', {}).get('topk', 64),
        max_samples=args.max_samples or analysis_config.get('max_samples', 1000),
        device=args.device or config.get('hardware', {}).get('device', 'cuda'),
    )


def cmd_train_gated_lora(args):
    """训练门控 LoRA"""
    import torch
    import json
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from datasets import load_from_disk, Audio
    from asr_tact.sae import SAE, SAEConfig
    from asr_tact.gated_lora import GatedLoRAConfig, GatedLoRAModel, select_gate_neurons
    
    config = load_config(args.config) if args.config else {}
    gated_lora_config = config.get('gated_lora', {})
    training_config = config.get('gated_lora_training', {})
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    output_dir = args.output_dir or os.path.join(
        config.get('output', {}).get('base_dir', './exp/asr_tact'),
        f'gated_lora-{timestamp}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    device = args.device or config.get('hardware', {}).get('device', 'cuda')
    
    print("=" * 70)
    print("ASR-TACT Gated LoRA Training")
    print("=" * 70)
    
    # 加载 SAE 模型
    print(f"Loading SAE checkpoint: {args.sae_checkpoint}")
    checkpoint = torch.load(args.sae_checkpoint, map_location='cpu')
    sae_config = SAEConfig.from_dict(checkpoint['config'])
    
    sae = SAE(
        input_dim=sae_config.input_dim,
        latent_dim=sae_config.latent_dim,
        norm_type=sae_config.norm_type,
        use_activate=sae_config.use_activate,
        topk_type=sae_config.topk_type,
        share_weight=sae_config.share_weight,
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    
    # 加载神经元分析结果
    print(f"Loading neuron analysis: {args.analysis_dir}")
    correlations_path = os.path.join(args.analysis_dir, 'neuron_correlations.json')
    with open(correlations_path, 'r') as f:
        neuron_correlations = json.load(f)
    
    stats_path = os.path.join(args.analysis_dir, 'neuron_stats.json')
    with open(stats_path, 'r') as f:
        neuron_stats = json.load(f)
    
    # 选择门控神经元
    neuron_importance = {
        int(k): v['activation_rate'] 
        for k, v in neuron_stats.items()
    }
    
    gate_neurons = select_gate_neurons(
        neuron_importance=neuron_importance,
        neuron_correlations={int(k): v for k, v in neuron_correlations.items()},
        top_n=gated_lora_config.get('num_gate_neurons', 64),
    )
    
    print(f"Selected {len(gate_neurons)} gate neurons")
    
    # 加载 Whisper 模型
    model_name = args.model_name or config.get('whisper', {}).get('model_name', 'openai/whisper-medium')
    print(f"Loading Whisper model: {model_name}")
    
    processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # 创建门控 LoRA 配置
    lora_config = GatedLoRAConfig(
        target_modules=gated_lora_config.get('target_modules', ['q_proj', 'v_proj']),
        lora_rank=gated_lora_config.get('lora_rank', 8),
        lora_alpha=gated_lora_config.get('lora_alpha', 16.0),
        lora_dropout=gated_lora_config.get('lora_dropout', 0.05),
        gate_threshold=gated_lora_config.get('gate_threshold', 0.5),
        gate_neurons={'default': gate_neurons},
    )
    
    # 保存配置
    lora_config.save(os.path.join(output_dir, 'gated_lora_config.json'))
    
    # 创建门控 LoRA 模型
    gated_model = GatedLoRAModel(
        base_model=whisper_model,
        sae=sae,
        config=lora_config,
        encoder_layer=config.get('sae', {}).get('encoder_layer', 12),
    )
    gated_model.to(device)
    gated_model.print_trainable_parameters()
    
    # 加载数据集
    dataset_path = args.dataset_path or config.get('dataset', {}).get('path', 'dataset/shanghai/shanghai_dataset')
    print(f"Loading dataset: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 数据预处理
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features[0]
        
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch
    
    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)
    
    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get('batch_size', 2),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config.get('learning_rate', 1e-5),
        warmup_steps=training_config.get('warmup_steps', 100),
        num_train_epochs=training_config.get('num_epochs', 10),
        eval_strategy="steps",
        eval_steps=training_config.get('eval_steps', 500),
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=3,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.get('hardware', {}).get('fp16', True),
        predict_with_generate=True,
        generation_max_length=225,
    )
    
    # 数据整理器
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            
            batch["labels"] = labels
            return batch
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=whisper_model.config.decoder_start_token_id,
    )
    
    # 创建训练器
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=gated_model.base_model,  # 使用包装后的模型
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    gated_model.save_lora_weights(os.path.join(output_dir, 'gated_lora_weights.pt'))
    processor.save_pretrained(output_dir)
    
    print("=" * 70)
    print("Training completed!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


def cmd_inference(args):
    """推理"""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa
    
    device = args.device or 'cuda'
    
    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    print(f"Processing audio: {args.audio_path}")
    audio, sr = librosa.load(args.audio_path, sr=16000)
    
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(input_features, max_length=225)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n" + "=" * 70)
    print("Transcription:")
    print(transcription)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="ASR-TACT: Targeted Activation Concept Tuning for ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SAE
  python run_asr_tact.py train_sae --config asr_tact/config.yaml
  
  # Analyze neurons
  python run_asr_tact.py analyze --sae_checkpoint exp/sae/best_model.pt
  
  # Train Gated LoRA
  python run_asr_tact.py train_gated_lora --sae_checkpoint exp/sae/best_model.pt --analysis_dir exp/neuron_analysis
  
  # Inference
  python run_asr_tact.py inference --model_path exp/gated_lora --audio_path test.wav
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # train_sae 子命令
    parser_sae = subparsers.add_parser('train_sae', help='Train SAE model')
    parser_sae.add_argument('--config', type=str, default='asr_tact/config.yaml', help='Config file path')
    parser_sae.add_argument('--model_name', type=str, help='Whisper model name')
    parser_sae.add_argument('--dataset_path', type=str, help='Dataset path')
    parser_sae.add_argument('--output_dir', type=str, help='Output directory')
    parser_sae.add_argument('--encoder_layer', type=int, help='Encoder layer')
    parser_sae.add_argument('--latent_dim', type=int, help='SAE latent dimension')
    parser_sae.add_argument('--topk', type=int, help='TopK for sparse activation')
    parser_sae.add_argument('--norm_type', type=str, help='Normalization type')
    parser_sae.add_argument('--batch_size', type=int, help='Batch size')
    parser_sae.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser_sae.add_argument('--learning_rate', type=float, help='Learning rate')
    parser_sae.add_argument('--device', type=str, help='Device')
    parser_sae.add_argument('--save_every', type=int, help='Save every N steps')
    parser_sae.add_argument('--log_every', type=int, help='Log every N steps')
    
    # analyze 子命令
    parser_analyze = subparsers.add_parser('analyze', help='Analyze neurons')
    parser_analyze.add_argument('--sae_checkpoint', type=str, required=True, help='SAE checkpoint path')
    parser_analyze.add_argument('--config', type=str, default='asr_tact/config.yaml', help='Config file path')
    parser_analyze.add_argument('--model_name', type=str, help='Whisper model name')
    parser_analyze.add_argument('--dataset_path', type=str, help='Dataset path')
    parser_analyze.add_argument('--output_dir', type=str, help='Output directory')
    parser_analyze.add_argument('--encoder_layer', type=int, help='Encoder layer')
    parser_analyze.add_argument('--topk', type=int, help='TopK')
    parser_analyze.add_argument('--max_samples', type=int, help='Max samples to analyze')
    parser_analyze.add_argument('--device', type=str, help='Device')
    
    # train_gated_lora 子命令
    parser_lora = subparsers.add_parser('train_gated_lora', help='Train Gated LoRA model')
    parser_lora.add_argument('--sae_checkpoint', type=str, required=True, help='SAE checkpoint path')
    parser_lora.add_argument('--analysis_dir', type=str, required=True, help='Neuron analysis directory')
    parser_lora.add_argument('--config', type=str, default='asr_tact/config.yaml', help='Config file path')
    parser_lora.add_argument('--model_name', type=str, help='Whisper model name')
    parser_lora.add_argument('--dataset_path', type=str, help='Dataset path')
    parser_lora.add_argument('--output_dir', type=str, help='Output directory')
    parser_lora.add_argument('--device', type=str, help='Device')
    
    # inference 子命令
    parser_infer = subparsers.add_parser('inference', help='Run inference')
    parser_infer.add_argument('--model_path', type=str, required=True, help='Model path')
    parser_infer.add_argument('--audio_path', type=str, required=True, help='Audio file path')
    parser_infer.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    if args.command == 'train_sae':
        cmd_train_sae(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'train_gated_lora':
        cmd_train_gated_lora(args)
    elif args.command == 'inference':
        cmd_inference(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
