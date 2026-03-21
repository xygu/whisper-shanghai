import mido
mid = mido.MidiFile('trial/solo.mid')
for i, track in enumerate(mid.tracks):
    print(f"Track {i}: {track.name}")
    for msg in track:
        print(msg)
