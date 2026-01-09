# 🧬 Federated BioNeMo Setup Guide

This guide details the exact steps to set up the NVIDIA BioNeMo environment on a Brev.dev cloud instance for the Federated Learning Hackathon.

## Prerequisites
1.  **NVIDIA NGC Account:** Sign up and generate an [API Key here](https://ngc.nvidia.com/setup/apikey).
2.  **Brev.dev Account:** Ensure you have a GPU instance (A10G or A100).
3. Or, access through the hackathon's brev platform.

---

##  Step 1: Launch Infrastructure (Brev)
1.  **Create Instance:**
    * Go to Brev Console.
    * **GPU:** Select `A10G` (Standard) or `A100` (High Performance).
    * **Disk Size:** **CRITICAL:** Set to **100GB** or more. The BioNeMo Docker image is ~15GB, and model checkpoints are huge.
2.  **Wait for Status:**
    * Wait until the status is `Running`.
    * *Troubleshooting:* If `nvidia-smi` fails or "VM Mode" fails, **delete and recreate** the instance.

---

## Step 2: VS Code & Repo Setup
1.  **Connect via VS Code:**
    * Run the command provided by Brev (e.g., `brev open <instance-name>`).
2.  **Setup Folder Structure (Host Side):**
    * Open the VS Code Terminal (Terminal > New Terminal).
    * Run the following to set up the repo:
    ```bash
    cd ~
    # Clone your repo (Use your actual repo name, e.g., FedProFit)
    cd ~
    git clone https://github.com/collaborativebioinformatics/FedProFit.git
    cd FedProFit
    
    # Create necessary data folders
    mkdir -p data/raw data/processed scripts notebooks configs
    ```

---

## Step 3: The Docker Plumbing (Critical)
BioNeMo lives inside a Docker container. We must "teleport" our code into it and fix the network permissions.

1.  **Login to NVIDIA Registry:**
    *(Recommended to ensure you can pull the image)*
    ```bash
    docker login nvcr.io
    # Username: $oauthtoken
    # Password: <YOUR_NGC_API_KEY>
    ```

2.  **Launch the Container:**
    Run this **exact** command in your VS Code terminal. 
    * **Note:** We use `--network=host` to fix the "blind container" issue so `pip install` works.
    
    ```bash
    docker run \
        --gpus all \
        --network=host \
        --ipc=host \
        -it --rm \
        --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v $(pwd):/workspace/project \
        nvcr.io/nvidia/clara/bionemo-framework:2.5 \
        bash
    ```
    * **Wait:** The first run will take ~5-10 mins to download the 15GB image.
    * **Success:** Your prompt will change to `root@<container-id>:/workspace/bionemo2#`.
    * Next also download nvflare for a federated workflow:
    ```bash
    pip install nvflare
    ```
---

##  Step 4: Dependency & Version Fixes
Once inside the container (`root@...`), run these commands to install NVFlare 2.7.1 and align dependencies.

1.  **Navigate to Project:**
    ```bash
    cd /workspace/project
    ```

2.  **Install Libraries (The "Dependency Dance"):**
    We strictly install NVFlare 2.7.1, but then downgrade Protobuf to keep BioNeMo (C++) from crashing.
    ```bash
    # 1. Install NVFlare 2.7.1
    pip install nvflare==2.7.1 pandas

    # 2. Downgrade Protobuf (Critical for BioNeMo stability)
    pip install protobuf==3.20.3
    ```
    * *Ignore the red warning text* about incompatibility. We verified this combination works.

3.  **Run the Smoke Test:**
    Verify that BioNeMo (the Brain) and NVFlare (the Coordinator) are talking to each other.
    ```bash
    python -c 'import torch; import nvflare; print("\n CRITICAL SUCCESS: System Operational")'
    ```


---

## 🛑 Cheat Sheet
* **Host Terminal (VS Code):** Used for Git commands (`git push`) and file editing.
* **Docker Terminal (`root@...`):** Used for running Python, Training, and NVFlare simulations.
* **Exiting Docker:** Type `exit` (Warning: This resets installed packages. You must run Step 4 again next time you launch).
* **BioNeMo at /workspace/bionemo2/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/finetune_esm2.py

-----

```markdown
# 🧬 Centralized ESM-2 Fine-Tuning: End-to-End Workflow

This document outlines the step-by-step process to replicate the centralized fine-tuning of the **ESM-2 (650M parameter)** model. It covers fetching the model weights, preparing the data splits, and running the training script within the NVIDIA BioNeMo container.

## 1. Model Setup: Fetching Weights

Before training, we must download the pre-trained model weights from NVIDIA's NGC catalog. We use a custom Python script to handle the download and placement of the checkpoint.

### **The Script: `fetch_model.py`**

Create a file named `fetch_model.py` in the root of your workspace (`/workspace/project/`) with the following content:

```python
from bionemo.core.data.load import load
import shutil
import os
import tarfile

print("⬇️  Attempting to fetch ESM-2 650M checkpoint...")

# 1. Download the official checkpoint from NGC
# The 'load' function handles authentication and caching automatically.
try:
    # This returns the path to the cached file (often a .nemo tar archive)
    cached_path = load("esm2/650m:2.0", source="ngc")
    print(f"✅ Downloaded to cache at: {cached_path}")
    
    # 2. Define the destination
    # We place it in the project root so it persists across container restarts.
    destination_nemo = "/workspace/project/esm2_650m.nemo"
    
    # 3. Handle the file
    # If the download is a .nemo file, copy it directly.
    # If it is a folder or needs processing, we handle it here.
    if os.path.exists(cached_path):
        print(f"📦 Copying to {destination_nemo}...")
        shutil.copy(cached_path, destination_nemo)
        
        # 4. Verify the file structure (Optional but recommended)
        # NeMo models are essentially tarballs. We verify it's a valid archive.
        if tarfile.is_tarfile(destination_nemo):
            print(f"🚀 Success! Valid .nemo archive ready at: {destination_nemo}")
        else:
            print(f"⚠️ Warning: File copied, but it might not be a valid tar archive.")
            
    else:
        print("❌ Error: Download fetched a path that doesn't exist.")

except Exception as e:
    print(f"❌ Failed to download. Error: {e}")

```

### **Execution**

Run the script inside the Docker container:

```bash
python fetch_model.py

```

### **File Locations**

* **Docker Side:** `/workspace/project/esm2_650m.nemo`
* **Host Side:** `/home/ubuntu/project/esm2_650m.nemo` (Assuming the project folder is mounted).
* **Why?** Keeping the model in the mounted volume ensures we don't have to re-download 5GB of data if the container is restarted.

---

## 2. Data Preparation: Creating Splits

We need to split our raw CSV files into train, val, and test sets. We use a script that automatically detects CSV files in the folder and creates the necessary structure.

### **The Script: `make_splits.py**`

Place this script inside your data folder (e.g., /workspace/project/data/). It works by scanning the directory it is located in.

Critical Requirements:

* The script must be in the same folder as your raw CSVs (human.csv, etc.).

* The script automatically renames mutated_sequence to sequences (required by BioNeMo).

**Command:**

```bash
cd /workspace/project/data
python make_splits.py

```
This will create a new folder called /workspace/project/data/splits/ containing your processed data.
---

## 3. Training: The Execution Script

Once the model is ready and data is split, we run the fine-tuning job. We use a shell script to manage the complex arguments required by BioNeMo.

### **The Script: `run_centralized_training.sh**`

Create this file in `/workspace/project/`:

```bash
#!/bin/bash

# Configuration
LR="1e-4"
STEPS="1000"           
BATCH_SIZE="8"

# 1. Pointing to local results (since the mount failed)
RESULT_DIR="./results/run_centralized_human"

# The Bionemo Script Path (Inside Docker)
TRAIN_SCRIPT="/workspace/bionemo2/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/finetune_esm2.py"

# Data Paths (Inside Docker)
TRAIN_DATA="/workspace/project/data/splits/human/train.csv"
VAL_DATA="/workspace/project/data/splits/human/val.csv"
CHECKPOINT="/workspace/project/esm2_650m.nemo"

echo " Starting Centralized Training..."
echo "   - Steps: $STEPS"
echo "   - Batch Size: $BATCH_SIZE"
echo "   - Output: $RESULT_DIR"

# Run command
python $TRAIN_SCRIPT \
    --train-data-path $TRAIN_DATA \
    --valid-data-path $VAL_DATA \
    --restore-from-checkpoint-path $CHECKPOINT \
    --task-type regression \
    --mlp-target-size 1 \
    --label-column DMS_score \
    --lr $LR \
    --micro-batch-size $BATCH_SIZE \
    --num-steps $STEPS \
    --num-gpus 1 \
    --result-dir $RESULT_DIR \
    --val-check-interval 50 \
    --log-every-n-steps 10 \
    --save-top-k 1 \
    --metric-to-monitor-for-checkpoints val_loss \
    --save-last-checkpoint \
    --avoid-ckpt-async-save

echo " Run Complete! Logs are in $RESULT_DIR"

```

### **Key Strategy Notes**

* **Storage Optimization:** We used `--save-top-k 1` to keep only the single best checkpoint (~5GB) instead of saving every epoch, which previously caused a 1TB overflow.
* **Resume Capability:** We added `--save-last-checkpoint`. This constantly overwrites a `last.ckpt` file, allowing us to resume training if the process is interrupted, without filling the disk.
* **Stability:** We included `--avoid-ckpt-async-save` to prevent threading crashes that can occur in containerized environments.

---

## 4. How to Reproduce

1. **Enter the Container:**
Ensure you are inside the BioNeMo docker container.
2. **Fetch Model:**
```bash
python fetch_model.py

```


3. **Run Training:**
```bash
chmod +x run_centralized_training.sh
./run_centralized_training.sh

```


4. **Verify:**
Check `./results/run_centralized_human` for logs and the final `.nemo` model.

```

```