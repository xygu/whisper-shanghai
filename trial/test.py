
# import torchaudio

# # 加载音频
# waveform, sample_rate = torchaudio.load("dataset/shanghai/WAV/A0002_S0003_0_G0003_G0004.wav")

# # 计算起始和结束采样点
# start_sample = 52 * sample_rate
# end_sample = 58 * sample_rate
# clip = waveform[:, start_sample:end_sample]

# # 保存为测试文件
# torchaudio.save("dataset/test_audio.wav", clip, sample_rate)
# print("音频片段已保存为 test_audio.wav")


import torch
from transformers import pipeline

pipeline = pipeline(
    task="automatic-speech-recognition",
    model="/mnt/workspace/workgroup/qq/ts/whisper/whisper-shanghai-final",
    torch_dtype=torch.float16,
    device=0
)
result = pipeline("dataset/test_audio.wav")
print(result)