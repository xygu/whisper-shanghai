
"""
上海话语音转录 Web 应用（Cascade 级联模式）
基于 Whisper 微调模型进行上海话语音识别，再通过 mT5 翻译为普通话

流程：上海话语音 -> Whisper ASR -> 上海话文本 -> mT5 翻译 -> 普通话文本

部署方式：
1. 本地运行: python app.py
2. HuggingFace Spaces: 上传此文件和模型文件
"""

import os

# ==================== Gradio Bug 修复 ====================
# 修复 Gradio 4.44.0 的 JSON schema 解析 bug
# 错误: TypeError: argument of type 'bool' is not iterable
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

try:
    import gradio_client.utils as _gradio_utils
    _original_json_schema_to_python_type = _gradio_utils._json_schema_to_python_type
    
    def _patched_json_schema_to_python_type(schema, defs=None):
        """修复：处理 schema 为 bool 的情况"""
        if isinstance(schema, bool):
            return "bool"
        return _original_json_schema_to_python_type(schema, defs)
    
    _gradio_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
except Exception:
    pass  # 如果补丁失败，继续运行

# ==================== HuggingFace 镜像配置（国内加速）====================
# 设置 HuggingFace 镜像，避免科学上网
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr
import torch
import numpy as np

# ==================== 配置 ====================
# 级联模式模型路径配置
# ASR 模型：上海话语音 -> 上海话文本
ASR_MODEL_PATH = os.environ.get(
    "ASR_MODEL_PATH",
    "./exp/whisper-shanghai-260318-004259"
)
# 翻译模型：上海话文本 -> 普通话文本
TRANSLATION_MODEL_PATH = os.environ.get(
    "TRANSLATION_MODEL_PATH",
    "./exp/translation-mt5-small-260320-231422/final_model"
)

# 检测设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# ==================== 全局模型变量 ====================
asr_model = None
asr_processor = None
translation_model = None
translation_tokenizer = None


def load_models():
    """
    加载级联模式所需的模型：
    1. Whisper ASR 模型：上海话语音 -> 上海话文本
    2. mT5 翻译模型：上海话文本 -> 普通话文本
    """
    global asr_model, asr_processor, translation_model, translation_tokenizer
    
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        MT5ForConditionalGeneration,
        MT5Tokenizer,
    )
    
    # ========== 加载 ASR 模型（上海话语音识别）==========
    print(f"[Cascade] 加载 ASR 模型: {ASR_MODEL_PATH}")
    print(f"[Cascade] 使用 HuggingFace 镜像: {os.environ.get('HF_ENDPOINT', '未设置')}")
    
    asr_processor = WhisperProcessor.from_pretrained(ASR_MODEL_PATH)
    asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_PATH)
    
    asr_model.to(DEVICE)
    asr_model.eval()
    
    # 清除可能冲突的配置
    if hasattr(asr_model.config, 'forced_decoder_ids'):
        asr_model.config.forced_decoder_ids = None
    if hasattr(asr_model.generation_config, 'forced_decoder_ids'):
        asr_model.generation_config.forced_decoder_ids = None
    
    print("✓ ASR 模型加载完成（上海话语音 -> 上海话文本）")
    
    # ========== 加载翻译模型（上海话 -> 普通话）==========
    print(f"[Cascade] 加载翻译模型: {TRANSLATION_MODEL_PATH}")
    
    if os.path.exists(TRANSLATION_MODEL_PATH):
        translation_tokenizer = MT5Tokenizer.from_pretrained(TRANSLATION_MODEL_PATH)
        translation_model = MT5ForConditionalGeneration.from_pretrained(TRANSLATION_MODEL_PATH)
        translation_model.to(DEVICE)
        translation_model.eval()
        print("✓ 翻译模型加载完成（上海话文本 -> 普通话文本）")
    else:
        # 尝试从 HuggingFace Hub 加载
        try:
            translation_tokenizer = MT5Tokenizer.from_pretrained(TRANSLATION_MODEL_PATH)
            translation_model = MT5ForConditionalGeneration.from_pretrained(TRANSLATION_MODEL_PATH)
            translation_model.to(DEVICE)
            translation_model.eval()
            print("✓ 翻译模型从 HuggingFace Hub 加载完成")
        except Exception as e:
            print(f"⚠️ 翻译模型加载失败: {e}")
            print("  级联模式需要翻译模型，请检查模型路径")


def resample_audio(audio_data, orig_sr, target_sr=16000):
    """重采样音频到目标采样率"""
    if orig_sr == target_sr:
        return audio_data
    
    try:
        import librosa
        return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # 简单的线性插值重采样
        duration = len(audio_data) / orig_sr
        new_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        return np.interp(indices, np.arange(len(audio_data)), audio_data)


def transcribe_shanghai(audio_data):
    """使用 Whisper 转录上海话"""
    try:
        # 检查音频数据有效性
        if audio_data is None or len(audio_data) == 0:
            return "(音频数据为空)"
        
        # 确保音频数据是正确的格式
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # 提取特征
        input_features = asr_processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)
        
        # 生成转录
        with torch.no_grad():
            predicted_ids = asr_model.generate(
                input_features,
                max_length=225,
                language="zh",
                task="transcribe",
            )
        
        # 解码结果
        transcription = asr_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        result = transcription.strip()
        return result if result else "(未识别到语音内容)"
        
    except Exception as e:
        print(f"转录错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"(转录失败: {str(e)})"


def translate_to_mandarin(shanghai_text):
    """将上海话文本翻译为普通话"""
    try:
        if translation_model is None or translation_tokenizer is None:
            return "(翻译模型未加载)"
        
        if not shanghai_text or shanghai_text.startswith("("):
            return "(无有效文本可翻译)"
        
        input_text = f"翻译上海话到普通话: {shanghai_text}"
        
        inputs = translation_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = translation_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        
        translated = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated if translated else "(翻译结果为空)"
        
    except Exception as e:
        print(f"翻译错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"(翻译失败: {str(e)})"


def process_audio_generator(audio_input):
    """
    处理音频输入，使用生成器实现动态状态更新
    
    Args:
        audio_input: Gradio 音频输入 (采样率, 音频数据) 或文件路径
    
    Yields:
        tuple: (上海话文本, 普通话文本, 状态信息)
    """
    if audio_input is None:
        yield "", "", "⚠️ 请上传或录制音频文件"
        return
    
    try:
        status = "🔄 正在处理音频...\n"
        yield "", "", status
        
        # 处理音频输入
        if isinstance(audio_input, tuple):
            sample_rate, audio_data = audio_input
            status += f"📊 采样率: {sample_rate} Hz\n"
            yield "", "", status
            
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            if sample_rate != 16000:
                status += f"🔄 重采样: {sample_rate} -> 16000 Hz\n"
                yield "", "", status
                audio_data = resample_audio(audio_data, sample_rate, 16000)
        
        elif isinstance(audio_input, str):
            status += f"📁 文件: {os.path.basename(audio_input)}\n"
            yield "", "", status
            
            try:
                import librosa
                audio_data, _ = librosa.load(audio_input, sr=16000, mono=True)
            except ImportError:
                import soundfile as sf
                audio_data, sr = sf.read(audio_input)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                if sr != 16000:
                    audio_data = resample_audio(audio_data, sr, 16000)
        else:
            yield "", "", f"❌ 不支持的音频输入类型: {type(audio_input)}"
            return
        
        duration = len(audio_data) / 16000
        status += f"⏱️ 音频时长: {duration:.1f} 秒\n"
        status += f"💻 设备: {DEVICE}\n"
        status += "─" * 30 + "\n"
        yield "", "", status
        
        # 转录上海话
        status += "🎙️ 正在转录上海话...（请稍候）\n"
        yield "", "", status
        
        shanghai_text = transcribe_shanghai(audio_data)
        status += f"✓ 上海话转录完成\n"
        yield shanghai_text, "", status
        
        # 翻译为普通话
        if translation_model is not None:
            status += "🔄 正在翻译为普通话...（请稍候）\n"
            yield shanghai_text, "", status
            
            mandarin_text = translate_to_mandarin(shanghai_text)
            status += f"✓ 普通话翻译完成\n"
        else:
            mandarin_text = "(翻译模型未加载)"
            status += "⚠️ 翻译模型未加载\n"
        
        status += "─" * 30 + "\n"
        status += "🎉 处理完成！"
        
        yield shanghai_text, mandarin_text, status
        
    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        yield "", "", error_msg


def create_demo():
    """创建 Gradio 界面"""
    
    # 自定义 CSS 样式
    custom_css = """
    /* 整体页面背景 */
    body {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0e6f6 50%, #e6f0f8 100%) !important;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        background: transparent !important;
    }
    
    /* 主面板背景 */
    .main-panel {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 24px !important;
        padding: 32px !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
    }
    
    /* 大圆角按钮 */
    .big-button {
        border-radius: 14px !important;
        padding: 16px 32px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        min-height: 52px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    
    .big-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* 主按钮样式 */
    .primary-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }
    
    .primary-btn:hover {
        background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%) !important;
    }
    
    /* 次要按钮样式 */
    .secondary-btn {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px solid #e0e0e0 !important;
        color: #666 !important;
    }
    
    .secondary-btn:hover {
        background: rgba(255, 255, 255, 1) !important;
        border-color: #d0d0d0 !important;
    }
    
    /* 音频组件样式 - 紧凑型 */
    .audio-compact {
        border-radius: 16px !important;
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        background: rgba(102, 126, 234, 0.05) !important;
        padding: 12px !important;
        min-height: 80px !important;
    }
    
    /* 录音按钮样式 - 红色圆圈 */
    .audio-compact button[aria-label*="record"],
    .audio-compact button[aria-label*="Record"],
    .audio-compact button[aria-label*="录"] {
        border-radius: 50% !important;
        width: 48px !important;
        height: 48px !important;
        background: #e53935 !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(229, 57, 53, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .audio-compact button[aria-label*="record"]:hover,
    .audio-compact button[aria-label*="Record"]:hover,
    .audio-compact button[aria-label*="录"]:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 16px rgba(229, 57, 53, 0.5) !important;
        background: #c62828 !important;
    }
    
    /* 录音中状态 - 红色方块（停止按钮） */
    .audio-compact button[aria-label*="stop"],
    .audio-compact button[aria-label*="Stop"],
    .audio-compact button[aria-label*="停"] {
        border-radius: 6px !important;
        width: 48px !important;
        height: 48px !important;
        background: #e53935 !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(229, 57, 53, 0.4) !important;
        animation: pulse 1.5s infinite !important;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(229, 57, 53, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(229, 57, 53, 0); }
        100% { box-shadow: 0 0 0 0 rgba(229, 57, 53, 0); }
    }
    
    /* 隐藏不必要的图标按钮 */
    .audio-compact button[aria-label*="edit"],
    .audio-compact button[aria-label*="Edit"],
    .audio-compact button[aria-label*="trim"],
    .audio-compact button[aria-label*="Trim"] {
        display: none !important;
    }
    
    /* 删除按钮样式 */
    .audio-compact button[aria-label*="clear"],
    .audio-compact button[aria-label*="Clear"],
    .audio-compact button[aria-label*="删"],
    .audio-compact button[aria-label*="remove"],
    .audio-compact button[aria-label*="Remove"] {
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        background: #757575 !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    .audio-compact button[aria-label*="clear"]:hover,
    .audio-compact button[aria-label*="Clear"]:hover,
    .audio-compact button[aria-label*="删"]:hover,
    .audio-compact button[aria-label*="remove"]:hover,
    .audio-compact button[aria-label*="Remove"]:hover {
        background: #616161 !important;
    }
    
    /* 音频图标改为白色 */
    .audio-compact svg {
        color: white !important;
        fill: white !important;
    }
    
    /* 文本框样式 */
    .result-textbox textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }
    
    /* 状态框样式 */
    .status-box textarea {
        border-radius: 12px !important;
        background: rgba(248, 249, 250, 0.9) !important;
        font-family: 'Monaco', 'Consolas', monospace !important;
        font-size: 12px !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 提示卡片 */
    .tip-card {
        background: rgba(102, 126, 234, 0.08) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        border-left: 3px solid #667eea !important;
        margin: 8px 0 !important;
    }
    
    /* 底部信息 */
    .footer-info {
        text-align: center;
        color: #888;
        font-size: 13px;
        padding: 16px;
    }
    
    /* 左侧面板 - 上移 */
    .left-panel {
        padding-top: 0 !important;
        padding-right: 24px !important;
    }
    
    /* 右侧面板 - 下移错开 */
    .right-panel {
        padding-top: 40px !important;
        padding-left: 24px !important;
    }
    
    /* 统一文本框字体大小 */
    .result-textbox textarea {
        font-size: 14px !important;
    }
    """
    
    with gr.Blocks(
        title="上海话语音转录",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        # 顶部标题区 - 更有质感
        gr.HTML(
            """
            <div style="text-align: center; padding: 40px 20px 30px;">
                <h1 style="
                    font-size: 2.4rem;
                    font-weight: 800;
                    margin-bottom: 12px;
                    text-shadow: 2px 2px 4px rgba(102, 126, 234, 0.2);
                    letter-spacing: 2px;
                ">
                    <span style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                    ">🎙️ 上海话语音转录系统</span>
                </h1>
                <p style="
                    color: #666; 
                    font-size: 15px; 
                    margin: 0;
                    font-weight: 400;
                    letter-spacing: 1px;
                ">
                    将上海话语音转录为文字，并翻译成普通话
                </p>
            </div>
            """
        )
        
        with gr.Row(equal_height=False):
            # 左侧：音频输入区
            with gr.Column(scale=1, elem_classes=["left-panel"]):
                gr.HTML('''
                    <h3 style="
                        margin-bottom: 12px; 
                        font-size: 15px;
                        font-weight: 600;
                    ">
                        <span style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                        ">音频输入</span>
                    </h3>
                ''')
                
                # 格式提示 - 放在输入区上方
                gr.HTML(
                    """
                    <div class="tip-card">
                        <span style="font-size: 12px; color: #555;">
                            🔴 点击红色圆形按钮开始录音，再次点击停止｜支持 WAV, MP3, M4A, FLAC, OGG
                        </span>
                    </div>
                    """
                )
                
                # 音频输入组件 - 支持录音和上传
                audio_input = gr.Audio(
                    type="numpy",
                    label="点击红色按钮开始录音，录制中点击方块停止",
                    sources=["microphone", "upload"],
                    elem_classes=["audio-compact"],
                    show_download_button=False,
                    show_share_button=False,
                )
                
                gr.HTML('<div style="height: 12px;"></div>')
                
                # 按钮区域 - 并排
                with gr.Row():
                    submit_btn = gr.Button(
                        "🚀 开始转录",
                        variant="primary",
                        size="lg",
                        elem_classes=["big-button", "primary-btn"],
                        scale=2,
                    )
                    clear_btn = gr.Button(
                        "🗑️ 重置",
                        variant="secondary",
                        size="lg",
                        elem_classes=["big-button", "secondary-btn"],
                        scale=1,
                    )
                
                gr.HTML('<div style="height: 16px;"></div>')
                
                # 状态显示
                gr.HTML('''
                    <h3 style="
                        margin-bottom: 8px; 
                        margin-top: 8px;
                        font-size: 15px;
                        font-weight: 600;
                    ">
                        <span style="
                            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                        ">处理状态</span>
                    </h3>
                ''')
                status_output = gr.Textbox(
                    label=None,
                    lines=6,
                    interactive=False,
                    elem_classes=["status-box"],
                    show_label=False,
                )
            
            # 右侧：转录结果区 - 错开布局
            with gr.Column(scale=1, elem_classes=["right-panel"]):
                gr.HTML('''
                    <h3 style="
                        margin-bottom: 12px; 
                        font-size: 15px;
                        font-weight: 600;
                    ">
                        <span style="
                            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                        ">转录结果</span>
                    </h3>
                ''')
                
                # 推理时长提示 - 放在转录区上方
                gr.HTML(
                    """
                    <div class="tip-card">
                        <span style="font-size: 12px; color: #555;">
                            ⏱️ CPU 推理约需 20-60 秒，请耐心等待
                        </span>
                    </div>
                    """
                )
                
                # 上海话输出
                gr.HTML('''
                    <p style="
                        margin: 0 0 6px 2px; 
                        font-size: 14px;
                        font-weight: 500;
                        color: #764ba2;
                    ">上海话转录结果</p>
                ''')
                shanghai_output = gr.Textbox(
                    label=None,
                    lines=5,
                    interactive=False,
                    elem_classes=["result-textbox"],
                    placeholder="转录后的上海话文本将显示在这里...",
                    show_label=False,
                )
                
                gr.HTML('<div style="height: 12px;"></div>')
                
                # 普通话输出
                gr.HTML('''
                    <p style="
                        margin: 0 0 6px 2px; 
                        font-size: 14px;
                        font-weight: 500;
                        color: #667eea;
                    ">普通话翻译结果</p>
                ''')
                mandarin_output = gr.Textbox(
                    label=None,
                    lines=5,
                    interactive=False,
                    elem_classes=["result-textbox"],
                    placeholder="翻译后的普通话文本将显示在这里...",
                    show_label=False,
                )
        
        # 底部信息
        gr.HTML(
            """
            <div style="
                text-align: center;
                padding: 20px;
                margin-top: 24px;
            ">
                <p style="color: #999; font-size: 12px; margin: 0;">
                    <span style="opacity: 0.8;">技术栈：Whisper + mT5 + Gradio</span>
                    <span style="margin: 0 8px; opacity: 0.4;">|</span>
                    <span style="opacity: 0.6;">使用 hf-mirror.com 国内镜像加速</span>
                </p>
            </div>
            """
        )
        
        # 绑定事件
        def handle_clear():
            """清除所有输入和输出"""
            return None, "", "", ""
        
        # 提交按钮事件 - 使用生成器实现动态更新
        submit_btn.click(
            fn=process_audio_generator,
            inputs=[audio_input],
            outputs=[shanghai_output, mandarin_output, status_output],
        )
        
        # 清除按钮事件
        clear_btn.click(
            fn=handle_clear,
            inputs=[],
            outputs=[audio_input, shanghai_output, mandarin_output, status_output],
        )
    
    return demo


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("上海话语音转录系统（级联模式 Cascade）")
    print("=" * 60)
    print(f"处理流程: 上海话语音 -> ASR -> 上海话文本 -> 翻译 -> 普通话文本")
    print(f"设备: {DEVICE}")
    print(f"HuggingFace 镜像: {os.environ.get('HF_ENDPOINT', '未设置')}")
    print(f"ASR 模型: {ASR_MODEL_PATH}")
    print(f"翻译模型: {TRANSLATION_MODEL_PATH}")
    print("=" * 60)
    
    # 加载模型
    load_models()
    
    # 创建并启动应用
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 远程服务器需要 share=True 或通过 IP 直接访问
        show_error=True,
        show_api=False,  # 禁用 API 文档，避免触发 Gradio 4.44.0 的 bug
    )
