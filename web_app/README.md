🎙️ 上海话语音转录 Web 应用（级联模式）

将上海话语音转录为文字，并翻译成普通话。

**处理流程：** 上海话语音 → Whisper ASR → 上海话文本 → mT5 翻译 → 普通话文本

## 📁 文件结构

```
web_app/
├── app.py              # 主应用程序（级联模式）
├── requirements.txt    # Python 依赖
└── README.md          # 本文档
```

## 🌐 HuggingFace 镜像加速

本应用已配置国内镜像加速，**无需科学上网**：

```python
# app.py 中已自动设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

如需手动设置环境变量：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 🖥️ 本地运行

```bash
cd web_app

# 安装依赖
pip install -r requirements.txt

# 运行应用
python app.py
```

然后访问 http://localhost:7860

## ☁️ 部署到 HuggingFace Spaces（免费）

### 步骤 1: 注册 HuggingFace 账号

访问 https://huggingface.co/join 注册账号（使用镜像：https://hf-mirror.com）

### 步骤 2: 上传模型到 HuggingFace Hub

由于模型文件较大，需要先上传到 HuggingFace Hub：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 设置镜像（国内加速，无需科学上网）
export HF_ENDPOINT=https://hf-mirror.com

# 登录 HuggingFace（需要 Access Token）
huggingface-cli login

# 上传 ASR 模型（级联模式：上海话语音 -> 上海话文本）
huggingface-cli upload your-username/whisper-shanghai ../exp/whisper-shanghai-260318-004259

# 上传翻译模型（级联模式：上海话文本 -> 普通话文本）
huggingface-cli upload your-username/mt5-shanghai-mandarin ../exp/translation-mt5-small-260320-231422/final_model
```

### 步骤 3: 创建 HuggingFace Space

1. 访问 https://huggingface.co/spaces（或镜像 https://hf-mirror.com/spaces）
2. 点击 **"Create new Space"**
3. 填写信息：
   - **Space name**: `shanghai-speech-recognition`
   - **License**: MIT
   - **SDK**: 选择 **Gradio**
   - **Hardware**: 选择 **CPU basic** (免费)
4. 点击 **"Create Space"**

### 步骤 4: 上传代码文件

在 Space 页面，点击 "Files" 标签，上传以下文件：

**app.py** - 需要修改模型路径：
```python
ASR_MODEL_PATH = "your-username/whisper-shanghai"
TRANSLATION_MODEL_PATH = "your-username/mt5-shanghai-mandarin"
```

**requirements.txt** - 直接上传即可

### 步骤 5: 等待部署

- 部署过程约需 5-10 分钟
- 部署完成后，你会得到一个公开的 URL
- 任何人都可以通过这个 URL 访问使用！

## ⚠️ 注意事项

| 项目 | 说明 |
|------|------|
| **模式** | 级联模式 (Cascade): ASR + 翻译 |
| **推理速度** | CPU 版约 20-60 秒/段音频 |
| **音频长度** | 建议 0.5-30 秒 |
| **支持格式** | WAV, MP3, M4A, FLAC, OGG 等 |
| **免费限制** | HuggingFace 免费版有使用配额 |
| **网络要求** | 已配置国内镜像，无需科学上网 |

## 🚀 升级到 GPU 版本

如果需要更快的推理速度（约 2-5 秒/段）：

1. **HuggingFace Spaces GPU**: 在 Settings 中选择 GPU 硬件
   - T4 small: $0.60/小时
   - A10G small: $1.05/小时

2. **自建服务器**: 部署到自己的 GPU 服务器
   - 修改 `demo.launch(share=True)` 生成公网链接

## 🔧 故障排除

**Q: 模型加载失败？**
- 检查模型路径是否正确
- 确认模型已上传到 HuggingFace Hub
- 确认已设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

**Q: 音频处理失败？**
- 确保音频格式正确（WAV/MP3/M4A 等）
- 检查音频是否损坏

**Q: 部署后页面空白？**
- 查看 Space 的 Logs 标签页
- 检查 requirements.txt 是否完整

**Q: 下载模型很慢？**
- 确认已设置 HuggingFace 镜像
- 镜像地址：https://hf-mirror.com
