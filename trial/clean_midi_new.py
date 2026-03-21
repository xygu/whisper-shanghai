import pretty_midi as pm, sys

# ─────────────── 参数 ───────────────
infile = 'trial/solo.mid'
outfile = 'trial/clean_solo.mid'
gap = 0.03
semitones = 13       # 触发截断的音程阈值（半音）

# ─────────────── 读取 MIDI ───────────────
m = pm.PrettyMIDI(infile)

# 删除 sustain pedal
for inst in m.instruments:
    inst.control_changes = [c for c in inst.control_changes if c.number != 64]

# 把所有音符汇总后按开始时间排序
notes = [n for inst in m.instruments for n in inst.notes]
notes.sort(key=lambda n: n.start)

active = []     # 当前仍在响的音符列表

for n in notes:
    # 1) 处理与新音重叠、且音程在 1.5 八度以内的旧音
    for a in active:
        if a.end >= n.start and abs(a.pitch - n.pitch) <= semitones:
            a.end = max(n.start - gap, a.start + 0.01)   # 至少留 10 ms 长度

    # 2) 把已结束的音移出 active
    active = [a for a in active if a.end > n.start]

    # 3) 把新音加入 active
    active.append(n)

# ─────────────── 写回文件 ───────────────
m.write(outfile)
print('saved', outfile)