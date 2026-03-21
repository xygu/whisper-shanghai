#!/usr/bin/env python3
"""
clean_piano.py  –  remove pedal and trim sustained notes

usage:
    python clean_piano.py in.mid out.mid   [--gap 0.03]
"""
import pretty_midi as pm, argparse, os, sys

infile = 'trial/solo.mid'
outfile = 'trial/clean_solo.mid'
gap = 0.03

midi = pm.PrettyMIDI(infile)

for inst in midi.instruments:
    # 1. throw the pedal away
    inst.control_changes = [c for c in inst.control_changes
                            if c.number != 64]

    # 2. trim overlapping notes (per pitch)
    notes_by_pitch = {}
# midi.write(outfile)

    for n in inst.notes:
        notes_by_pitch.setdefault(n.pitch, []).append(n)

    for pitch_notes in notes_by_pitch.values():
        pitch_notes.sort(key=lambda n: n.start)
        for i, n in enumerate(pitch_notes[:-1]):
            nxt = pitch_notes[i+1]
            new_end = min(n.end, nxt.start - gap)
            # keep the note at least 10 ms long
            n.end = max(new_end, n.start + 0.01)

midi.write(outfile)
print(f"written {outfile}")

