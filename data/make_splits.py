import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# Get the folder where this script is currently sitting (which is now /workspace/project/data)
current_script_dir = Path(__file__).parent.resolve()

# Since the script is IN the data folder, the source is the same directory
source_dir = current_script_dir
output_base = current_script_dir / "splits"

print(f"Script location: {current_script_dir}")
print(f"Looking for CSVs in: {source_dir}")
print(f"Saving splits to: {output_base}")

# Get list of all CSV files in the source directory
# We filter for .csv to ignore directories like 'splits' or python scripts
files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
print(f"Found {len(files)} files: {files}")

for file in files:
    # CLEANUP NAME: "human_files.csv" -> "human"
    taxon_name = file.replace('_files.csv', '').replace('.csv', '') 
    
    file_path = os.path.join(source_dir, file)
    
    # 1. Read Data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Skipped {file}: Could not read CSV. Error: {e}")
        continue

    print(f"\nProcessing {taxon_name} (Total rows: {len(df)})")
    
    # 2. Split Data (80% Train, 10% Val, 10% Test)
    try:
        # First split: Train (80%) vs Temp (20%)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        # Second split: Val (50% of Temp) vs Test (50% of Temp)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    except ValueError as e:
         print(f"Skipped {taxon_name}: Not enough data to split. Error: {e}")
         continue
    
    # 3. Create Output Directory
    save_dir = output_base / taxon_name
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. Save Splits
    train_df.to_csv(save_dir / "train.csv", index=False)
    val_df.to_csv(save_dir / "val.csv", index=False)
    test_df.to_csv(save_dir / "test.csv", index=False)
    
    print(f"Saved to {save_dir}/ [train.csv, val.csv, test.csv]")

print("\nAll splits completed!")