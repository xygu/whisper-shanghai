from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

model_path = "pretrained_models/whisper-medium"
language = "Chinese"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, language=language, task="transcribe")

print('a')

# import librosa
# import matplotlib.pyplot as plt

# audio, sr = librosa.load('dataset/test_audio.wav')
# audio = librosa.resample(audio, orig_sr = sr, target_sr = 16000)
# features = feature_extractor(audio, sampling_rate = 16000)
# plt.imshow(features['input_features'][0,:,:600], aspect='auto', origin='lower')
# plt.title('Mel Spectrogram Features')
# plt.xlabel('Frame Index')
# plt.ylabel('Mel Frequency Band')
# # plt.colorbar()
# plt.savefig('dataset/test_audio1.png')

# print('b')
 