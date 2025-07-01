from datasets import load_dataset, Audio, Dataset
import json
import os
import logging

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为DEBUG
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
)



sr = 16000
root_dir = 'dataset/shanghai'
# 读取我们刚准备的数据
logging.info('opening data...')
with open(os.path.join(root_dir,"whisper_finetune_data_short.jsonl"), "r", encoding="utf8") as f:
    lines = [json.loads(line) for line in f]

logging.info('tranferring to Hugging Face Dataset...')
dataset = Dataset.from_list(lines)
logging.info('casting columns...')
dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))

# 加载模型和分词器
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# model_name = "openai/whisper-small"
model_name = "pretrained_models/transformer_small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d"
logging.info('loading processor...')
processor = WhisperProcessor.from_pretrained(model_name)
logging.info('loading model...')
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 数据预处理
def preprocess(example):
    audio = example["audio"]
    input_features = processor(audio["array"], sampling_rate=sr, return_tensors="pt").input_features[0]
    labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids[0]
    example["input_features"] = input_features
    example["labels"] = labels
    return example
logging.info('loading dataset...')
dataset = dataset.map(preprocess)

logging.info('importing traner module...')
from transformers import TrainingArguments, Trainer
logging.info('loading training args...')
training_args = TrainingArguments(
    output_dir="./whisper-shanghai",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    fp16=True,
    report_to="none",
)
logging.info('setting up trainer...')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)

logging.info('start training...')
trainer.train()