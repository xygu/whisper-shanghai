from datasets import load_dataset, Audio, Dataset
import json

# 读取我们刚准备的数据
with open("whisper_finetune_data.jsonl", "r", encoding="utf8") as f:
    lines = [json.loads(line) for line in f]

# 转成 Hugging Face Dataset
dataset = Dataset.from_list(lines)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 加载模型和分词器
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 数据预处理
def preprocess(example):
    audio = example["audio"]
    input_features = processor(audio["array"], sampling_rate=16000, return_tensors="pt").input_features[0]
    labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids[0]
    example["input_features"] = input_features
    example["labels"] = labels
    return example

dataset = dataset.map(preprocess)

# 训练参数
from transformers import TrainingArguments, Trainer

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()