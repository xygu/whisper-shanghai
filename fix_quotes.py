#!/usr/bin/env python3
"""修复 generate_neuron_explanation_prompt 中的中文引号语法错误"""

filepath = '/mnt/workspace/workgroup/qq/ts/whisper/asr_tact/run_full_pipeline.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复 char_dist_lines 里的中文引号
old1 = (
    '        char_dist_lines.append(\n'
    '            f"  {rank:2d}. \u5b57=\u201c{char}\u201d  \u51fa\u73b0={stat[\'count\']}\u6b21  "\n'
    '            f"\u5e73\u5747\u6fc0\u6d3b={stat[\'mean_activation\']:.3f}  \u4e0a\u4e0b\u6587\u793a\u4f8b: {context_examples}"\n'
    '        )'
)
new1 = (
    '        char_dist_lines.append(\n'
    '            f"  {rank:2d}. \u5b57=[{char}]  \u51fa\u73b0={stat[\'count\']}\u6b21  "\n'
    '            f"\u5e73\u5747\u6fc0\u6d3b={stat[\'mean_activation\']:.3f}  \u4e0a\u4e0b\u6587\u793a\u4f8b: {context_examples}"\n'
    '        )'
)

# 修复 token_evidence_lines 里的中文引号
old2 = (
    '        token_evidence_lines.append(\n'
    '            f"  {i:2d}. \u6fc0\u6d3b={activation:.3f}  \u5b57=\u201c{char}\u201d  \u4e0a\u4e0b\u6587=\u201c{context}\u201d  \u6574\u53e5=\u201c{transcript}\u201d"\n'
    '        )'
)
new2 = (
    '        token_evidence_lines.append(\n'
    '            f"  {i:2d}. \u6fc0\u6d3b={activation:.3f}  \u5b57=[{char}]  \u4e0a\u4e0b\u6587=[{context}]  \u6574\u53e5=[{transcript}]"\n'
    '        )'
)

changed = False
if old1 in content:
    content = content.replace(old1, new1, 1)
    print("Fixed char_dist_lines quotes OK")
    changed = True
else:
    # 尝试直接字符串替换（中文引号可能有编码差异）
    content = content.replace(
        'f"  {rank:2d}. \u5b57=\u201c{char}\u201d  \u51fa\u73b0={stat[\'count\']}\u6b21  "',
        'f"  {rank:2d}. \u5b57=[{char}]  \u51fa\u73b0={stat[\'count\']}\u6b21  "',
        1
    )
    print("Tried inline replace for char_dist_lines")
    changed = True

if old2 in content:
    content = content.replace(old2, new2, 1)
    print("Fixed token_evidence_lines quotes OK")
    changed = True
else:
    content = content.replace(
        'f"  {i:2d}. \u6fc0\u6d3b={activation:.3f}  \u5b57=\u201c{char}\u201d  \u4e0a\u4e0b\u6587=\u201c{context}\u201d  \u6574\u53e5=\u201c{transcript}\u201d"',
        'f"  {i:2d}. \u6fc0\u6d3b={activation:.3f}  \u5b57=[{char}]  \u4e0a\u4e0b\u6587=[{context}]  \u6574\u53e5=[{transcript}]"',
        1
    )
    print("Tried inline replace for token_evidence_lines")
    changed = True

if changed:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("File written.")

# 验证语法
import ast
try:
    ast.parse(content)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")
