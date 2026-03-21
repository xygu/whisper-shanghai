from music21 import converter, instrument, note, chord

# 加载 MIDI 文件
midi_file = 'trial/solo.mid'
score = converter.parse(midi_file)


score.write('musicxml', fp='trial/solo.xml')
