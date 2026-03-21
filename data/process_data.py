import os
import re
import json
from pathlib import Path

def arrange_wav(base_path):
    """
    Process the Shanghai dialect dataset and organize it into a structured format.
    
    Args:
        base_path (str): Base path to the dataset directory
        
    Returns:
        list: List of dictionaries containing the organized data
    """
    
    # Define paths
    wav_path = os.path.join(base_path, "WAV")
    txt_path = os.path.join(base_path, "TXT")
    txt_cn_path = os.path.join(base_path, "TXT_CN")
    
    # Get all TXT files
    txt_files = [f for f in os.listdir(txt_path) if f.endswith('.txt')]
    
    data_list = []
    
    for txt_file in txt_files:
        # Get file name without extension
        file_name = os.path.splitext(txt_file)[0]
        
        # Define paths for this specific file
        txt_file_path = os.path.join(txt_path, txt_file)
        txt_cn_file_path = os.path.join(txt_cn_path, txt_file)
        wav_file_path = os.path.join(wav_path, file_name + ".wav")
        
        # Check if corresponding files exist
        if not os.path.exists(txt_cn_file_path):
            print(f"Warning: Missing Chinese transcription file for {txt_file}")
            continue
            
        if not os.path.exists(wav_file_path):
            print(f"Warning: Missing WAV file for {txt_file}")
            continue
        
        # Read the TXT file (Shanghai dialect)
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            txt_lines = f.readlines()
            
        # Read the TXT_CN file (Mandarin Chinese)
        with open(txt_cn_file_path, 'r', encoding='utf-8') as f:
            txt_cn_lines = f.readlines()
            
        # Create a dictionary to map time ranges to Chinese sentences
        cn_transcription_map = {}
        for line in txt_cn_lines:
            line = line.strip()
            if line:
                # Extract time range and sentence
                match = re.match(r'\[([\d.]+),([\d.]+)\]\s+(G\d+)\s+(male|female)\s+(.*)', line)
                if match:
                    start_time, end_time, _,_,sentence = match.groups()
                    time_key = f"[{start_time},{end_time}]"
                    cn_transcription_map[time_key] = sentence
        
        # Process each line in the TXT file
        for line in txt_lines:
            line = line.strip()
            if line:
                # Extract information using regex
                match = re.match(r'\[([\d.]+),([\d.]+)\]\s+(G\d+)\s+(male|female)\s+(.*)', line)
                if match:
                    start_time, end_time, speaker_id, gender, sentence_sh = match.groups()
                    
                    # Create time key to find corresponding Chinese transcription
                    time_key = f"[{start_time},{end_time}]"
                    sentence_cn = cn_transcription_map.get(time_key, "")
                    
                    # Create data dictionary
                    data_entry = {
                        "audio": wav_file_path,
                        "sentence_sh": sentence_sh,
                        "sentence_cn": sentence_cn,
                        "gender": gender,
                        "speaker_id": speaker_id,
                        "start_time": float(start_time),
                        "end_time": float(end_time)
                    }
                    
                    data_list.append(data_entry)
                    
    return data_list

# Example usage
if __name__ == "__main__":
    # Define the base path to your dataset
    base_path = "/mnt/workspace/workgroup/qq/ts/whisper/dataset/shanghai"
    
    # Process the dataset
    data = arrange_wav(base_path)
    
    # Print first few entries as example
    for i, entry in enumerate(data[:3]):
        print(f"Entry {i+1}:")
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        print()
    
    # Optionally save to a JSON file
    with open("shanghai_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(data)} entries total.")