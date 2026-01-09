import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
current_script_dir = Path(__file__).parent.resolve()
source_dir = current_script_dir
output_base = current_script_dir / "splits"

print(f"Script location: {current_script_dir}")
print(f"Looking for CSVs in: {source_dir}")

files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
print(f"Found {len(files)} files: {files}")

for file in files:
    taxon_name = file.replace('_files.csv', '').replace('.csv', '') 
    file_path = os.path.join(source_dir, file)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Skipped {file}: {e}")
        continue

    # --- THE FIX: RENAME COLUMN AUTOMATICALLY ---
    # BioNeMo strictly requires the column to be named 'sequence'
    if 'mutated_sequence' in df.columns:
        print(f"   🔧 Renaming 'mutated_sequence' -> 'sequences' for {taxon_name}")
        df.rename(columns={'mutated_sequence': 'sequences'}, inplace=True)
    # --------------------------------------------

    print(f"Processing {taxon_name} (Total rows: {len(df)})")
    
    try:
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    except ValueError as e:
         print(f"Skipped {taxon_name}: {e}")
         continue
    
    save_dir = output_base / taxon_name
    os.makedirs(save_dir, exist_ok=True)
    
    train_df.to_csv(save_dir / "train.csv", index=False)
    val_df.to_csv(save_dir / "val.csv", index=False)
    test_df.to_csv(save_dir / "test.csv", index=False)
    
    print(f"   ✅ Saved to splits/{taxon_name}/")

print("\nAll splits completed and column names fixed!")