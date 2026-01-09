# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import os
import glob
import shutil
import sys

# Import filters from the 'custom' package
from custom.bionemo_filters import BioNeMoParamsFilter, BioNeMoStateDictFilter

from nvflare import FilterType
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_common.widgets.decomposer_reg import DecomposerRegister
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import BaseScriptRunner


def main(args):
    # Define Client Mapping (Name -> Data Folder)
    CLIENT_MAP = [
        {"name": "site-1", "dataset": "human"},
        {"name": "site-2", "dataset": "eukaryote"},       
        # {"name": "site-3", "dataset": "prokaryote"}, 
        # {"name": "site-4", "dataset": "virus"}       
    ]
    
    # Create BaseFedJob
    job = BaseFedJob(name=f"{args.exp_name}_scl_esm2_{args.model}")

    # Define the controller
    controller = FedAvg(
        num_clients=len(CLIENT_MAP),
        num_rounds=args.num_rounds,
    )
    job.to_server(controller)
    job.to_server(DecomposerRegister(["nvflare.app_opt.pt.decomposers.TensorDecomposer"]))

    # Use Local Model Checkpoint
    checkpoint_path = "/workspace/project/esm2_650m.nemo"
    print(f"Using Model Checkpoint: {checkpoint_path}")

    # Loop through specific clients
    for client_info in CLIENT_MAP:
        client_name = client_info["name"]
        dataset_name = client_info["dataset"]

        # Point to your specific data splits
        train_data_path = f"/workspace/project/data/splits/{dataset_name}/train.csv"
        val_data_path = f"/workspace/project/data/splits/{dataset_name}/val.csv"

        # Validation Logic
        if args.num_rounds > 1:
            val_check_interval = args.local_steps
        else:
            val_check_interval = int(args.local_steps / 2)

        # DYNAMIC ARGUMENTS
        precision = "16-mixed"
        
        # Determine dataset class based on task type
        if args.task_type == "regression":
            dataset_class = "InMemorySingleValueDataset"
        else:
            dataset_class = "InMemoryProteinDataset" 
        
        script_args = (
            f"--restore-from-checkpoint-path {checkpoint_path} "
            f"--train-data-path {train_data_path} "
            f"--valid-data-path {val_data_path} "
            f"--config-class ESM2FineTuneSeqConfig "
            f"--dataset-class {dataset_class} "            
            f"--task-type {args.task_type} "               
            f"--label-column {args.label_column} "         
            f"--mlp-target-size {args.target_size} "       
            f"--mlp-ft-dropout 0.1 "
            f"--mlp-hidden-size 256 "
            f"--experiment-name {job.name} "
            f"--num-steps {args.local_steps} "
            f"--num-gpus 1 "
            f"--val-check-interval {val_check_interval} "
            f"--log-every-n-steps 10 "
            f"--lr {args.lr} "                              
            f"--result-dir bionemo "
            f"--micro-batch-size {args.batch_size} "       
            f"--precision {precision} "
            f"--save-top-k 1 "
            f"--limit-val-batches {args.limit_val_batches} "
            f"--save-last-checkpoint "
        )

        if args.encoder_frozen:
            script_args += "--encoder-frozen "
        
        print(f" Configuring {client_name} -> {dataset_name} (Task: {args.task_type})")

        # Define training script runner
        runner = BaseScriptRunner(
            script=f"custom/{args.train_script}",
            launch_external_process=True,
            framework="pytorch",
            server_expected_format="pytorch",
            launcher=SubprocessLauncher(
                script=f"python3 custom/{args.train_script} {script_args}", 
                launch_once=False, 
                shutdown_timeout=100.0
            ),
        )
        job.to(runner, client_name)
        job.to(
            BioNeMoParamsFilter(precision), client_name, tasks=["train", "validate"], filter_type=FilterType.TASK_DATA
        )
        job.to(BioNeMoStateDictFilter(), client_name, tasks=["train", "validate"], filter_type=FilterType.TASK_RESULT)
        job.to(DecomposerRegister(["nvflare.app_opt.pt.decomposers.TensorDecomposer"]), client_name)

    # Export Paths
    export_dir = "/workspace/project/federated/jobs"
    workspace_dir = f"/workspace/project/federated/workspaces/{job.name}"
    job_folder_path = os.path.join(export_dir, job.name)

    # 1. Export the Job Configuration
    job.export_job(export_dir)
    print(f"✅ Job exported to {export_dir}")

    # 2. MANUAL COPY OF CUSTOM FILES
    # We copy files into the exported job folder on disk.
    print("📦 Copying 'custom/' files to job directories...")
    
    local_custom_dir = "custom"
    # Target every app folder inside the job
    targets = ["app_server"] + [f"app_{c['name']}" for c in CLIENT_MAP]

    for target_app in targets:
        dest_dir = os.path.join(job_folder_path, target_app, "custom")
        os.makedirs(dest_dir, exist_ok=True)
        
        for file_path in glob.glob(f"{local_custom_dir}/*.py"):
            shutil.copy(file_path, dest_dir)
            
    print(f"   - Copied custom scripts to: {targets}")
    
    # 3. RUN SIMULATOR VIA CLI
    # [FIX] We use os.system to run the CLI. This forces the simulator to use 
    # the folder we just prepared on disk (job_folder_path), ensuring the custom files are found.
    print("\n🚀 Starting Simulator...")
    
    # Construct the simulator command
    # -w: workspace directory
    # -n: number of clients
    # -t: number of threads (we use 1 usually, or equal to clients)
    # -gpu: gpu indices
    cmd = (
        f"nvflare simulator {job_folder_path} "
        f"-w {workspace_dir} "
        f"-n {len(CLIENT_MAP)} "
        f"-t {len(CLIENT_MAP)} " 
        f"-gpu {args.sim_gpus}"
    )
    
    print(f"   Command: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Basic Sim Args
    parser.add_argument("--num_rounds", type=int, help="Number of rounds", required=False, default=3)
    parser.add_argument("--local_steps", type=int, help="Steps per round", required=False, default=10)
    parser.add_argument("--train_script", type=str, help="Training script", required=False, default="finetune_esm2_federated.py")
    parser.add_argument("--exp_name", type=str, help="Job name prefix", required=False, default="fed_dms")
    parser.add_argument("--model", type=str, help="ESM2 model", required=False, default="650m")
    parser.add_argument("--sim_gpus", type=str, required=False, default="0")

    # --- NEW ARGS PASSED FROM BASH SCRIPT ---
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro batch size")
    parser.add_argument("--task_type", type=str, default="regression", help="Task type")
    parser.add_argument("--label_column", type=str, default="DMS_score", help="Column for labels")
    parser.add_argument("--target_size", type=int, default=1, help="MLP target size")
    parser.add_argument("--encoder-frozen", action="store_true", help="Freeze encoder")
    parser.add_argument("--limit-val-batches", type=float, default=1.0, help="Limit validation batches (1.0 = all)")

    args = parser.parse_args()
    args.num_clients = 0 

    main(args)