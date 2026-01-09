from bionemo.core.data.load import load
import shutil
import os

print("⬇️  Attempting to fetch ESM-2 650M checkpoint...")

# This command asks BioNeMo to download the official checkpoint from NGC
# It handles the URL and authentication automatically.
try:
    checkpoint_path = load("esm2/650m:2.0", source="ngc")
    print(f"✅ Downloaded to cache at: {checkpoint_path}")
    
    # Copy it to your current folder so it's easy to find
    destination = "/workspace/project/esm2_650m.nemo"
    if os.path.exists(checkpoint_path):
        shutil.copy(checkpoint_path, destination)
        print(f"🚀  Success! Copied to: {destination}")
    else:
        print("❌ Error: Download fetched a path that doesn't exist.")

except Exception as e:
    print(f"❌ Failed to download. Error: {e}")
