from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import librosa

audio, sr = librosa.load('trial/solo.mp3', sr=sample_rate)

# Transcriptor
transcriptor = PianoTranscription(device='cuda')    # 'cuda' | 'cpu'

# Transcribe and write out to MIDI file
transcribed_dict = transcriptor.transcribe(audio, 'trial/solo.mid')

print('finish')