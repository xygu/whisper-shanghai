"""
Parse llm_prompts.md and generate one JSON summary file per Neuron token.
Output: exp/sae_config_C/llm_prompts/result/{neuron_id}.json
"""

import re
import json
import os

INPUT_FILE = "/mnt/workspace/workgroup/qq/ts/whisper/exp/sae_config_C/llm_prompts/llm_prompts.md"
OUTPUT_DIR = "/mnt/workspace/workgroup/qq/ts/whisper/exp/sae_config_C/llm_prompts/result"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_char_distribution(block_text):
    """Parse the Character Distribution section."""
    chars = []
    pattern = re.compile(
        r'(\d+)\.\s+字=\[(.+?)\]\s+出现=(\d+)次\s+平均激活=([\d.]+)\s+上下文示例:\s*(.+)'
    )
    for match in pattern.finditer(block_text):
        rank = int(match.group(1))
        char = match.group(2)
        count = int(match.group(3))
        mean_activation = float(match.group(4))
        context_examples = [c.strip() for c in match.group(5).split('、')]
        chars.append({
            "rank": rank,
            "char": char,
            "count": count,
            "mean_activation": mean_activation,
            "context_examples": context_examples
        })
    return chars


def parse_top30_tokens(block_text):
    """Parse the Top 30 Token Evidence section."""
    tokens = []
    pattern = re.compile(
        r'(\d+)\.\s+激活=([\d.]+)\s+字=\[(.+?)\]\s+上下文=\[(.+?)\]\s+整句=\[(.+?)\]'
    )
    for match in pattern.finditer(block_text):
        tokens.append({
            "rank": int(match.group(1)),
            "activation": float(match.group(2)),
            "char": match.group(3),
            "context": match.group(4),
            "full_sentence": match.group(5)
        })
    return tokens


def parse_acoustic_features(block_text):
    """Parse the Acoustic Features Statistics section."""
    features = {}
    pattern = re.compile(r'-\s+(\w+):\s+mean=([\d.]+),\s+std=([\d.]+),\s+n=(\d+)')
    for match in pattern.finditer(block_text):
        features[match.group(1)] = {
            "mean": float(match.group(2)),
            "std": float(match.group(3)),
            "n": int(match.group(4))
        }
    return features


def parse_linguistic_features(block_text):
    """Parse the Linguistic Features Statistics section."""
    features = {}
    # Numeric features: mean + std
    num_pattern = re.compile(r'-\s+(\w+):\s+mean=([\d.]+),\s+std=([\d.]+)')
    for match in num_pattern.finditer(block_text):
        features[match.group(1)] = {
            "mean": float(match.group(2)),
            "std": float(match.group(3))
        }
    # Boolean features: X/30 (Y% True)
    bool_pattern = re.compile(r'-\s+(\w+):\s+(\d+)/(\d+)\s+\(([\d.]+)%\s+True\)')
    for match in bool_pattern.finditer(block_text):
        features[match.group(1)] = {
            "true_count": int(match.group(2)),
            "total": int(match.group(3)),
            "true_ratio": float(match.group(4)) / 100.0
        }
    return features


def split_neuron_blocks(content):
    """Split the full markdown content into per-neuron blocks."""
    # Split on the separator line before each neuron block
    separator = "======================================================================"
    # Find all neuron header positions
    neuron_header_pattern = re.compile(
        r'CURRENT CONCEPT DATA: Neuron #(\d+)\s*\n\(Total activation events: (\d+)\)'
    )
    blocks = []
    matches = list(neuron_header_pattern.finditer(content))
    for index, match in enumerate(matches):
        neuron_id = int(match.group(1))
        total_events = int(match.group(2))
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        block_text = content[start:end]
        blocks.append((neuron_id, total_events, block_text))
    return blocks


def parse_neuron_block(neuron_id, total_events, block_text):
    """Parse a single neuron block into a structured summary dict."""
    # Extract sections by their markdown headers
    char_dist_match = re.search(
        r'### Character Distribution.*?\n(.*?)(?=###|\Z)', block_text, re.DOTALL
    )
    top30_match = re.search(
        r'### Top 30 Token Evidence.*?\n(.*?)(?=###|\Z)', block_text, re.DOTALL
    )
    acoustic_match = re.search(
        r'### Acoustic Features Statistics.*?\n(.*?)(?=###|\Z)', block_text, re.DOTALL
    )
    linguistic_match = re.search(
        r'### Linguistic Features Statistics.*?\n(.*?)(?=###|={10,}|\Z)', block_text, re.DOTALL
    )

    char_distribution = parse_char_distribution(char_dist_match.group(1)) if char_dist_match else []
    top30_tokens = parse_top30_tokens(top30_match.group(1)) if top30_match else []
    acoustic_features = parse_acoustic_features(acoustic_match.group(1)) if acoustic_match else {}
    linguistic_features = parse_linguistic_features(linguistic_match.group(1)) if linguistic_match else {}

    return {
        "neuron_id": neuron_id,
        "total_activation_events": total_events,
        "character_distribution_top10": char_distribution,
        "top30_tokens": top30_tokens,
        "acoustic_features_statistics": acoustic_features,
        "linguistic_features_statistics": linguistic_features
    }


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file_handle:
        content = file_handle.read()

    blocks = split_neuron_blocks(content)
    print(f"Found {len(blocks)} neuron blocks.")

    for neuron_id, total_events, block_text in blocks:
        summary = parse_neuron_block(neuron_id, total_events, block_text)
        output_path = os.path.join(OUTPUT_DIR, f"{neuron_id}.json")
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(summary, out_file, ensure_ascii=False, indent=2)
        print(f"  Saved Neuron #{neuron_id} -> {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
