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
