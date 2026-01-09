# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import glob
import inspect
import os
import shutil

from custom.bionemo_filters import BioNeMoParamsFilter, BioNeMoStateDictFilter

from nvflare import FilterType
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_common.widgets.decomposer_reg import DecomposerRegister
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.script_runner import BaseScriptRunner





def main(args):
    CLIENT_MAP = [
        {"name": "site-1", "dataset": "human"},
        {"name": "site-2", "dataset": "eukaryote"},
        # {"name": "site-3", "dataset": "prokaryote"},
        # {"name": "site-4", "dataset": "virus"},
    ]

    job = BaseFedJob(name=f"{args.exp_name}_scl_esm2_{args.model}")

    # Server: controller + decomposer
    controller = FedAvg(
    num_clients=len(CLIENT_MAP),
    num_rounds=args.num_rounds,
    allow_empty_global_weights=True,
    )
    job.to_server(controller)
    job.to_server(DecomposerRegister(["nvflare.app_opt.pt.decomposers.TensorDecomposer"]))
    

    # Server: global model persistor (saved model lives under server workspace)
    job.to_server(
        PTFileModelPersistor(global_model_file_name="esm2_aggregated_model.pt"),
        id="persistor",
    )

    # Initial checkpoint for round 0 (clients start from this)
    checkpoint_path = args.initial_nemo_ckpt
    print(f"Using initial NeMo checkpoint: {checkpoint_path}")

    # Workspace + export paths
    export_dir = args.export_dir
    workspace_dir = os.path.join(args.workspace_root, job.name)
    job_folder_path = os.path.join(export_dir, job.name)

    # Add clients
    for client_info in CLIENT_MAP:
        client_name = client_info["name"]
        dataset_name = client_info["dataset"]

        train_data_path = f"/workspace/project/data/splits/{dataset_name}/train.csv"
        val_data_path = f"/workspace/project/data/splits/{dataset_name}/val.csv"

        # Validate only at end of each round for clean per-round metrics
        val_check_interval = args.local_steps

        # Dataset class for regression/classification
        if args.task_type == "regression":
            dataset_class = "InMemorySingleValueDataset"
        else:
            dataset_class = "InMemoryProteinDataset"

        # Put results under the NVFlare workspace so you can always find them
        result_dir = os.path.join(workspace_dir, "bionemo_outputs", client_name)

        precision = args.precision

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
            f"--log-every-n-steps {args.log_every_n_steps} "
            f"--lr {args.lr} "
            f"--result-dir {result_dir} "
            f"--micro-batch-size {args.batch_size} "
            f"--precision {precision} "
            f"--save-top-k 0 "
            f"--limit-val-batches {args.limit_val_batches} "
            f"--save-last-checkpoint "
            
        )

        if args.encoder_frozen:
            script_args += "--encoder-frozen "
        if args.create_tensorboard_logger:
            script_args += "--create-tensorboard-logger "

        print(f"Configuring {client_name} -> {dataset_name} (task={args.task_type})")
        print(f"Training args: {script_args}")

        runner = BaseScriptRunner(
            script=f"custom/{args.train_script}",
            launch_external_process=True,
            framework="pytorch",
            server_expected_format="pytorch",
            launcher=SubprocessLauncher(
                script=f"python3 custom/{args.train_script} {script_args}",
                launch_once=False,
                shutdown_timeout=100.0,
            ),
        )

        job.to(runner, client_name)

        job.to(
            BioNeMoParamsFilter(precision),
            client_name,
            tasks=["train", "validate"],
            filter_type=FilterType.TASK_DATA,
        )
        job.to(
            BioNeMoStateDictFilter(),
            client_name,
            tasks=["train", "validate"],
            filter_type=FilterType.TASK_RESULT,
        )
        job.to(DecomposerRegister(["nvflare.app_opt.pt.decomposers.TensorDecomposer"]), client_name)

    # Export job
    job.export_job(export_dir)
    print(f"✅ Job exported to {export_dir}")

    # Copy custom scripts into exported job folders (server + clients)
    print("📦 Copying 'custom/' files into exported job apps...")
    local_custom_dir = "custom"
    targets = ["app_server"] + [f"app_{c['name']}" for c in CLIENT_MAP]

    for target_app in targets:
        dest_dir = os.path.join(job_folder_path, target_app, "custom")
        os.makedirs(dest_dir, exist_ok=True)
        for file_path in glob.glob(f"{local_custom_dir}/*.py"):
            shutil.copy(file_path, dest_dir)

    print(f"   - Copied custom scripts to: {targets}")

    # Run simulator
    print("\n🚀 Starting Simulator...")
    cmd = (
        f"nvflare simulator {job_folder_path} "
        f"-w {workspace_dir} "
        f"-n {len(CLIENT_MAP)} "
        f"-t {len(CLIENT_MAP)} "
        f"-gpu {args.sim_gpus}"
    )
    print(f"Command: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # FL controls
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--local_steps", type=int, default=200)

    # scripts
    parser.add_argument("--train_script", type=str, default="finetune_esm2_federated.py")
    parser.add_argument("--exp_name", type=str, default="fed_dms")
    parser.add_argument("--model", type=str, default="650m")
    parser.add_argument("--sim_gpus", type=str, default="0")

    # paths
    parser.add_argument("--export_dir", type=str, default="/workspace/project/federated/jobs")
    parser.add_argument("--workspace_root", type=str, default="/workspace/project/federated/workspaces")
    parser.add_argument("--initial_nemo_ckpt", type=str, default="/workspace/project/esm2_650m.nemo")

    # training args
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--task_type", type=str, default="regression")
    parser.add_argument("--label_column", type=str, default="DMS_score")
    parser.add_argument("--target_size", type=int, default=1)
    parser.add_argument("--encoder_frozen", action="store_true")

    # validation/logging
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    # precision
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--create_tensorboard_logger", action="store_true")

    args = parser.parse_args()
    main(args)
