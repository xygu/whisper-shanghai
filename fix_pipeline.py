#!/usr/bin/env python3
"""修复 run_full_pipeline.py：恢复 Step 5，修复 Step 6"""

filepath = '/mnt/workspace/workgroup/qq/ts/whisper/asr_tact/run_full_pipeline.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# ── 1. 恢复 Step 5 ──────────────────────────────────────────────────────────
old_step5 = (
    '    # 对每个神经元，按激活值降序排列，取前 30 条作为 top_tokens，前 500 条做 char 频率统计\n'
    '    neuron_top_tokens = {}\n'
    '    TOP_TOKENS_COUNT = 30\n'
    '    STAT_TOKENS_COUNT = 500\n'
    '    CHAR_DIST_TOP_N = 10\n'
    '\n'
    '    for nid in top_10_neurons:\n'
    '        tokens = neuron_token_activations[nid]\n'
    '        tokens.sort(key=lambda x: x[\'activation\'], reverse=True)\n'
    '        total_token_count = len(tokens)\n'
    '\n'
    '        # 前 30 条：完整保留每条 token 的 char 信息\n'
    '        top_tokens = []\n'
    '        for token_record in tokens[:TOP_TOKENS_COUNT]:\n'
    '            alignment = token_record.get(\'alignment\', {})\n'
    '            top_tokens.append({\n'
    '                \'aligned_char\': alignment.get(\'aligned_char\', \'\'),\n'
    '                \'aligned_context\': alignment.get(\'aligned_context\', \'\'),\n'
    '                \'transcript\': token_record[\'transcript\'],\n'
    '                \'activation\': token_record[\'activation\'],\n'
    '                \'time_sec\': alignment.get(\'time_sec\', 0.0),\n'
    '                \'sample_idx\': token_record[\'sample_idx\'],\n'
    '                \'acoustic_features\': token_record.get(\'acoustic_features\', {}),\n'
    '                \'linguistic_features\': token_record.get(\'linguistic_features\', {}),\n'
    '            })\n'
    '\n'
    '        # 前 500 条：统计 char 频率，取出现次数最多的前 10 个字\n'
    '        char_stats = defaultdict(lambda: {\'count\': 0, \'activation_sum\': 0.0, \'contexts\': []})\n'
    '        for token_record in tokens[:STAT_TOKENS_COUNT]:\n'
    '            alignment = token_record.get(\'alignment\', {})\n'
    '            char = alignment.get(\'aligned_char\', \'\')\n'
    '            context = alignment.get(\'aligned_context\', \'\')\n'
    '            if not char:\n'
    '                continue\n'
    '            char_stats[char][\'count\'] += 1\n'
    '            char_stats[char][\'activation_sum\'] += token_record[\'activation\']\n'
    '            # 每个字最多保留 5 个不重复的上下文示例\n'
    '            if context and context not in char_stats[char][\'contexts\'] and len(char_stats[char][\'contexts\']) < 5:\n'
    '                char_stats[char][\'contexts\'].append(context)\n'
    '\n'
    '        # 按出现次数排序，取前 10 个字\n'
    '        sorted_chars = sorted(char_stats.items(), key=lambda x: x[1][\'count\'], reverse=True)\n'
    '        char_distribution = {}\n'
    '        for char, stat in sorted_chars[:CHAR_DIST_TOP_N]:\n'
    '            char_distribution[char] = {\n'
    '                \'count\': stat[\'count\'],\n'
    '                \'mean_activation\': round(stat[\'activation_sum\'] / stat[\'count\'], 4),\n'
    '                \'contexts\': stat[\'contexts\'],\n'
    '            }\n'
    '\n'
    '        neuron_top_tokens[nid] = {\n'
    '            \'top_tokens\': top_tokens,\n'
    '            \'char_distribution\': char_distribution,\n'
    '            \'total_token_count\': total_token_count,\n'
    '        }\n'
    '        print(f"Neuron {nid}: {total_token_count} total tokens, top {TOP_TOKENS_COUNT} saved, "\n'
    '              f"char_distribution top {len(char_distribution)} chars (from {min(STAT_TOKENS_COUNT, total_token_count)} tokens)")'
)

new_step5 = (
    '    # 对每个神经元，取激活值最大的前 top_percent token\n'
    '    neuron_top_tokens = {}\n'
    '    top_percent = getattr(args, \'top_percent\', 1.0)\n'
    '    \n'
    '    for nid in top_10_neurons:\n'
    '        tokens = neuron_token_activations[nid]\n'
    '        tokens.sort(key=lambda x: x[\'activation\'], reverse=True)\n'
    '        top_count = max(1, int(len(tokens) * top_percent))\n'
    '        neuron_top_tokens[nid] = tokens[:top_count]\n'
    '        print(f"Neuron {nid}: {len(tokens)} total tokens, top {top_percent*100:.0f}% = {top_count} tokens")'
)

if old_step5 in content:
    content = content.replace(old_step5, new_step5, 1)
    print("Step 5 restored OK")
else:
    print("ERROR: Step 5 old block not found")

# ── 2. 修复 Step 6 cmd_generate_prompt 中的语法错误块 ──────────────────────
# 当前文件中 prompts = {} 被删掉了，for 循环缩进也乱了
old_step6 = (
    "    top_10_neurons = data['top_10_neurons']\n"
    "    neuron_top_tokens = data['neuron_top_tokens']\n"
    "        for nid in top_10_neurons:\n"
    "        nid_str = str(nid)\n"
    "        neuron_data = neuron_top_tokens.get(nid_str, {})\n"
    "\n"
    "        # 兼容新结构（dict）和旧结构（list）\n"
    "        if isinstance(neuron_data, dict):\n"
    "            top_tokens = neuron_data.get('top_tokens', [])\n"
    "            char_distribution = neuron_data.get('char_distribution', {})\n"
    "            total_token_count = neuron_data.get('total_token_count', 0)\n"
    "        else:\n"
    "            # 旧格式兼容：list of token records\n"
    "            top_tokens = neuron_data[:30]\n"
    "            char_distribution = {}\n"
    "            total_token_count = len(neuron_data)\n"
    "\n"
    "        if not top_tokens and not char_distribution:\n"
    "            continue\n"
    "\n"
    "        # 从 top 30 条 token 中收集声学/语言特征统计\n"
    "        acoustic_stats = defaultdict(list)\n"
    "        linguistic_stats = defaultdict(list)\n"
    "\n"
    "        for token in top_tokens[:30]:\n"
    "            acoustic = token.get('acoustic_features', {})\n"
    "            linguistic = token.get('linguistic_features', {})\n"
    "\n"
    "            for k, v in acoustic.items():\n"
    "                if isinstance(v, (int, float)):\n"
    "                    acoustic_stats[k].append(v)\n"
    "\n"
    "            for k, v in linguistic.items():\n"
    "                if isinstance(v, (int, float, bool)):\n"
    "                    linguistic_stats[k].append(v)\n"
    "\n"
    "        # 生成 prompt\n"
    "        prompt = generate_neuron_explanation_prompt(\n"
    "            nid,\n"
    "            top_tokens,\n"
    "            char_distribution,\n"
    "            total_token_count,\n"
    "            acoustic_stats,\n"
    "            linguistic_stats,\n"
    "        )        \n"
    "        prompts[nid] = prompt"
)

new_step6 = (
    "    top_10_neurons = data['top_10_neurons']\n"
    "    neuron_top_tokens = data['neuron_top_tokens']\n"
    "\n"
    "    prompts = {}\n"
    "\n"
    "    for nid in top_10_neurons:\n"
    "        nid_str = str(nid)\n"
    "        tokens = neuron_top_tokens.get(nid_str, [])\n"
    "\n"
    "        if not tokens:\n"
    "            continue\n"
    "\n"
    "        # 声学/语言特征统计：取前 30 条整句\n"
    "        acoustic_stats = defaultdict(list)\n"
    "        linguistic_stats = defaultdict(list)\n"
    "\n"
    "        for t in tokens[:30]:\n"
    "            acoustic = t.get('acoustic_features', {})\n"
    "            linguistic = t.get('linguistic_features', {})\n"
    "\n"
    "            for k, v in acoustic.items():\n"
    "                if isinstance(v, (int, float)):\n"
    "                    acoustic_stats[k].append(v)\n"
    "\n"
    "            for k, v in linguistic.items():\n"
    "                if isinstance(v, (int, float, bool)):\n"
    "                    linguistic_stats[k].append(v)\n"
    "\n"
    "        # char 频率统计：前 500 条，取 freq 前 10\n"
    "        char_stats = defaultdict(lambda: {'count': 0, 'activation_sum': 0.0, 'contexts': []})\n"
    "        for t in tokens[:500]:\n"
    "            alignment = t.get('alignment', {})\n"
    "            char = alignment.get('aligned_char', '')\n"
    "            context = alignment.get('aligned_context', '')\n"
    "            if not char:\n"
    "                continue\n"
    "            char_stats[char]['count'] += 1\n"
    "            char_stats[char]['activation_sum'] += t.get('activation', 0.0)\n"
    "            if context and context not in char_stats[char]['contexts'] and len(char_stats[char]['contexts']) < 5:\n"
    "                char_stats[char]['contexts'].append(context)\n"
    "\n"
    "        sorted_chars = sorted(char_stats.items(), key=lambda x: x[1]['count'], reverse=True)\n"
    "        char_distribution = {}\n"
    "        for char, stat in sorted_chars[:10]:\n"
    "            char_distribution[char] = {\n"
    "                'count': stat['count'],\n"
    "                'mean_activation': round(stat['activation_sum'] / stat['count'], 4),\n"
    "                'contexts': stat['contexts'],\n"
    "            }\n"
    "\n"
    "        # top 30 token 证据列表（以 char 为标识，保留 aligned_context 和整句）\n"
    "        top_tokens_evidence = []\n"
    "        for t in tokens[:30]:\n"
    "            alignment = t.get('alignment', {})\n"
    "            top_tokens_evidence.append({\n"
    "                'aligned_char': alignment.get('aligned_char', ''),\n"
    "                'aligned_context': alignment.get('aligned_context', ''),\n"
    "                'transcript': t.get('transcript', ''),\n"
    "                'activation': t.get('activation', 0.0),\n"
    "            })\n"
    "\n"
    "        total_token_count = len(tokens)\n"
    "\n"
    "        # 生成 prompt\n"
    "        prompt = generate_neuron_explanation_prompt(\n"
    "            nid,\n"
    "            top_tokens_evidence,\n"
    "            char_distribution,\n"
    "            total_token_count,\n"
    "            acoustic_stats,\n"
    "            linguistic_stats,\n"
    "        )\n"
    "\n"
    "        prompts[nid] = prompt"
)

if old_step6 in content:
    content = content.replace(old_step6, new_step6, 1)
    print("Step 6 loop fixed OK")
else:
    print("ERROR: Step 6 old block not found")
    # 打印周边内容帮助定位
    idx = content.find("neuron_top_tokens = data['neuron_top_tokens']")
    if idx >= 0:
        print("Found anchor at char", idx)
        print(repr(content[idx:idx+300]))

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done.")
