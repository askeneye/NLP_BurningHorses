import os
from datasets import load_dataset
from pathlib import Path

# From: nlp_burninghorses/MIKKE/scripts/utils/download_conll-2003-data.py
# Levels: 1 (utils) -> 2 (scripts) -> 3 (MIKKE) -> 4 (nlp_burninghorses)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "conll2003"

def setup_conll():
    print(f"🚀 Downloading CoNLL-2003 to: {RAW_DATA_PATH}")
    
    # Create the directory if it doesn't exist
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # 2. Load from Hugging Face
    # CoNLL-2003 is a standard benchmark used in the OADA paper 
    dataset = load_dataset("lhoestq/conll2003")
    
    # 3. Save as JSON or CSV for easy "raw" access
    for split in ['train', 'validation', 'test']:
        file_path = RAW_DATA_PATH / f"{split}.json"
        dataset[split].to_json(file_path)
        print(f"✅ Saved {split} split to {file_path}")

if __name__ == "__main__":
    setup_conll()