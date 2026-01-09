# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied and adapted for NVFlare from https://github.com/NVIDIA/bionemo-framework/blob/main/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/finetune_esm2.py

import shutil
from pathlib import Path
from typing import Optional

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.dataset import InMemoryProteinDataset, InMemorySingleValueDataset
from bionemo.esm2.model.finetune.sequence_model import ESM2FineTuneSeqConfig

# Resue parser and config constants from bionemo
from bionemo.esm2.scripts.finetune_esm2 import get_parser
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.model.config import TorchmetricsConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, RichModelSummary
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule

# (1) import nvflare lightning client API
import nvflare.client.lightning as flare
import nvflare.client.api as flare_api

def train_model(
    # ... keep all your existing arguments exactly the same ...
    train_data_path: Path,
    valid_data_path: Path,
    num_nodes: int,
    devices: int,
    min_seq_length: Optional[int],
    max_seq_length: int,
    result_dir: Path,
    num_steps: int,
    limit_val_batches: int,
    val_check_interval: int,
    log_every_n_steps: Optional[int],
    num_dataset_workers: int,
    lr: float,
    micro_batch_size: int,
    accumulate_grad_batches: int,
    experiment_name: str,
    resume_if_exists: bool,
    precision: PrecisionTypes,
    task_type: str = "regression",
    encoder_frozen: bool = False,
    scale_lr_layer: Optional[str] = None,
    lr_multiplier: float = 1.0,
    mlp_ft_dropout: float = 0.25,
    mlp_hidden_size: int = 256,
    mlp_target_size: int = 1,
    cnn_dropout: float = 0.25,
    cnn_hidden_size: int = 32,
    cnn_num_classes: int = 3,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = False,
    wandb_tags: Optional[list[str]] = None,
    wandb_group: Optional[str] = None,
    wandb_id: Optional[str] = None,
    wandb_anonymous: Optional[bool] = False,
    wandb_log_model: bool = False,
    pipeline_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    create_tensorboard_logger: bool = False,
    restore_from_checkpoint_path: Optional[str] = None,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    nsys_profiling: bool = False,
    nsys_start_step: int = 0,
    nsys_end_step: Optional[int] = None,
    nsys_ranks: list[int] = [0],
    dataset_class: type[InMemoryProteinDataset] = InMemorySingleValueDataset,
    config_class: type[BioBertConfig] = ESM2FineTuneSeqConfig,
    metric_tracker: Callback | None = None,
    overlap_grad_reduce: bool = False,
    overlap_param_gather: bool = True,
    average_in_collective: bool = True,
    grad_reduce_in_fp32: bool = False,
    label_column: str = "labels",
    classes: list[str] = None,
) -> tuple[Path, Callback | None, nl.Trainer]:

    # Create the result directory if it does not exist.
    result_dir.mkdir(parents=True, exist_ok=True)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            average_in_collective=average_in_collective,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=True,
        ),
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=False,
        ckpt_parallel_load=True,
    )

    wandb_config: Optional[WandbConfig] = (
        None
        if wandb_project is None
        else WandbConfig(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags,
            group=wandb_group,
            id=wandb_id,
            anonymous=wandb_anonymous,
            log_model=wandb_log_model,
        )
    )

    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        nl_callbacks.PreemptionCallback(),
    ]
    if metric_tracker is not None:
        callbacks.append(metric_tracker)
    if nsys_profiling:
        if nsys_end_step is None:
            nsys_end_step = num_steps
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=nsys_start_step, end_step=nsys_end_step, ranks=nsys_ranks, gen_shape=True
            )
        )

    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        num_nodes=num_nodes,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision=precision,
            params_dtype=get_autocast_dtype(precision),
            pipeline_dtype=get_autocast_dtype(precision),
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            autocast_enabled=False,
        ),
    )

    # ---------------------------------------------------------
    # CHANGE 1: REMOVED flare.patch(trainer)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # CHANGE 2: Manual Receive
    # ---------------------------------------------------------
    import torch 
    input_model = flare.receive()
    print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")

    # Add TB streamer
    from custom.bionemo_tb_streamer import BioNeMoTBStreamer
    trainer.callbacks.append(BioNeMoTBStreamer(start_step=input_model.current_round * num_steps))

    # ---------------------------------------------------------
    # CHANGE 3: REMOVED manually appending FLCallback
    # ---------------------------------------------------------

    # Directory management
    keep_last_ckpt_only = True
    if keep_last_ckpt_only:
        previous_ckpt_dir = (
            result_dir / f"round{input_model.current_round - 1}" / experiment_name / "dev" / "checkpoints"
        )
        if previous_ckpt_dir.is_dir():
            print(f"Removing previous checkpoint directory {previous_ckpt_dir}")
            shutil.rmtree(previous_ckpt_dir)

    result_dir = result_dir / f"round{input_model.current_round}"

    # LR Scheduling
    if input_model.current_round > 0:
        lr_step_reduce = 1.05
        new_lr = lr / (input_model.current_round * lr_step_reduce)
        new_lr_multiplier = lr_multiplier / (input_model.current_round * lr_step_reduce)
        print(f"Reduce lr {lr} by {input_model.current_round * lr_step_reduce}: {new_lr}")
    else:
        new_lr = lr
        new_lr_multiplier = lr_multiplier

    tokenizer = get_tokenizer()

    train_dataset = dataset_class.from_csv(train_data_path, task_type=task_type, label_column=label_column)
    valid_dataset = dataset_class.from_csv(valid_data_path, task_type=task_type, label_column=label_column)
    if task_type == "classification" and classes:
         train_dataset.label_tokenizer.build_vocab([classes])

    # ---------------------------------------------------------
    # FIX from LLM: Calculate global_batch_size manually
    # Global BS = Micro BS * Devices (GPUs) * Nodes * Grad Accumulation
    # ---------------------------------------------------------
    global_batch_size = micro_batch_size * devices * num_nodes * accumulate_grad_batches

    data_module = ESM2FineTuneDataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        num_workers=num_dataset_workers,
        tokenizer=tokenizer,
    )

    # Metrics
    train_metric = None
    if task_type == "regression":
        valid_metric = TorchmetricsConfig(class_path="MeanSquaredError", task="regression", metric_name="val_mse")
    else:
        valid_metric = TorchmetricsConfig(
            class_path="Accuracy",
            task="classification",
            kwargs={
                "task": "multiclass",
                "threshold": 0.5,
                "num_classes": data_module.train_dataset.label_tokenizer.vocab_size,
            },
            metric_name="val_acc",
        )

    config = config_class(
        task_type=task_type,
        encoder_frozen=encoder_frozen,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=str(restore_from_checkpoint_path),
        initial_ckpt_skip_keys_with_these_prefixes=[f"{task_type}_head"],
        train_metric=train_metric,
        valid_metric=valid_metric,
    )

    task_dependent_attr = {
        "mlp_ft_dropout": mlp_ft_dropout,
        "mlp_hidden_size": mlp_hidden_size,
        "mlp_target_size": mlp_target_size,
        "cnn_dropout": cnn_dropout,
        "cnn_hidden_size": cnn_hidden_size,
        "cnn_num_classes": cnn_num_classes,
    }
    for attr, value in task_dependent_attr.items():
        if hasattr(config, attr):
            setattr(config, attr, value)

    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=new_lr,
            optimizer="adam",
            use_distributed_optimizer=True,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
        ),
    )
    if scale_lr_layer:
        optimizer.scale_lr_cond = lambda name, param: scale_lr_layer in name
        optimizer.lr_mult = new_lr_multiplier

    # Create the Lightning Module
    module = biobert_lightning_module(config=config, tokenizer=tokenizer, optimizer=optimizer)

    # ---------------------------------------------------------
    # CHANGE 4: Manually Load Global Weights into Module
    # ---------------------------------------------------------
    if input_model.params:
        print(f" Loading {len(input_model.params)} layers from Global Model...")
        # Convert params to CPU tensors
        incoming_state = {k: torch.as_tensor(v) for k, v in input_model.params.items()}
        
        # Load logic (handles potential prefix mismatches if necessary)
        missing, unexpected = module.load_state_dict(incoming_state, strict=False)
        print(f"   Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print("No params received (Round 0). Using initial checkpoint weights.")

    save_local_ckpt = False
    if save_local_ckpt:
        checkpoint_callback = nl_callbacks.ModelCheckpoint(
            save_last=save_last_checkpoint,
            monitor=metric_to_monitor_for_checkpoints,
            save_top_k=save_top_k,
            every_n_train_steps=val_check_interval,
            always_save_context=True,
            filename="checkpoint-{step}-{consumed_samples}",
        )
    else:
        checkpoint_callback = None

    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_config=wandb_config,
        ckpt_callback=checkpoint_callback,
    )

    # Perform Training
    llm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=None,
    )

    # ---------------------------------------------------------
    # CHANGE 5: Manually Extract and Send
    # ---------------------------------------------------------
    print(" Preparing result for Server...")
    
    # Extract weights to CPU numpy/tensors
    output_state_dict = {
        k: v.cpu().numpy() 
        for k, v in module.state_dict().items() 
        if v is not None
    }
    
    # Calculate Val Loss for aggregation (optional, but good practice)
    val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))

    output_model = flare_api.FLModel(
        params=output_state_dict,
        metrics={"val_loss": val_loss}
    )

    #flare.send(output_model) # COmmenting out to avoid failure of flare.send() i.e. for lightning import
    flare_api.send(output_model)
    print(" Model sent to server manually.")

    if checkpoint_callback:
        ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    else:
        ckpt_path = None
    return ckpt_path, metric_tracker, trainer


def finetune_esm2_entrypoint():
    """Entrypoint for running ESM2 finetuning."""
    flare.init()
    # 1. get arguments
    parser = get_parser()

    # Add some FL specific arguments
    parser.add_argument(
        "--classes",
        type=str,
        required=False,
        default=None,
        help="Unique strings describing the classes for classification. Used to build the same label vocabulary on each client. Should be comma separate list of strings, e.g. 'pos,neg'",
    )
    args = parser.parse_args()

    if args.classes:
        if args.task_type != "classification":
            parser.error("Use --classes argument only with --task-type 'classification'")
        classes = args.classes.split(",")
    else:
        classes = None

    # ---------------------------------------------------------
    # CRITICAL FIX: Convert String to Class
    # ---------------------------------------------------------
    # The command line argument comes in as a string (e.g. "InMemorySingleValueDataset")
    # We must convert it to the actual Class object.
    
    if isinstance(args.dataset_class, str):
        if args.dataset_class == "InMemorySingleValueDataset":
            actual_dataset_class = InMemorySingleValueDataset
        elif args.dataset_class == "InMemoryProteinDataset":
            actual_dataset_class = InMemoryProteinDataset
        else:
            # Fallback or error if an unknown string is passed
            print(f"Warning: Unknown dataset_class string '{args.dataset_class}'. Defaulting to InMemorySingleValueDataset.")
            actual_dataset_class = InMemorySingleValueDataset
    else:
        # If it's already a class (some parsers handle this, but rare via CLI)
        actual_dataset_class = args.dataset_class
    # ---------------------------------------------------------

    # to avoid padding for single value labels:
    if args.min_seq_length is not None and actual_dataset_class is InMemorySingleValueDataset:
        parser.error("Arguments --min-seq-length cannot be set when using InMemorySingleValueDataset.")

    # 2. Call pretrain with args
    train_model(
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        result_dir=args.result_dir,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        wandb_group=args.wandb_group,
        wandb_id=args.wandb_id,
        wandb_anonymous=args.wandb_anonymous,
        wandb_log_model=args.wandb_log_model,
        wandb_offline=args.wandb_offline,
        num_steps=args.num_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        num_dataset_workers=args.num_dataset_workers,
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        task_type=args.task_type,
        encoder_frozen=args.encoder_frozen,
        scale_lr_layer=args.scale_lr_layer,
        lr_multiplier=args.lr_multiplier,
        # single value classification / regression mlp
        mlp_ft_dropout=args.mlp_ft_dropout,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_target_size=args.mlp_target_size,
        # token-level classification cnn
        cnn_dropout=args.cnn_dropout,
        cnn_hidden_size=args.cnn_hidden_size,
        cnn_num_classes=args.cnn_num_classes,
        experiment_name=args.experiment_name,
        resume_if_exists=args.resume_if_exists,
        restore_from_checkpoint_path=args.restore_from_checkpoint_path,
        save_last_checkpoint=args.save_last_checkpoint,
        metric_to_monitor_for_checkpoints=args.metric_to_monitor_for_checkpoints,
        save_top_k=args.save_top_k,
        nsys_profiling=args.nsys_profiling,
        nsys_start_step=args.nsys_start_step,
        nsys_end_step=args.nsys_end_step,
        nsys_ranks=args.nsys_ranks,
        dataset_class=actual_dataset_class,  # <--- CHANGED HERE
        config_class=args.config_class,
        overlap_grad_reduce=args.overlap_grad_reduce,
        overlap_param_gather=not args.no_overlap_param_gather,
        average_in_collective=not args.no_average_in_collective,
        grad_reduce_in_fp32=args.grad_reduce_in_fp32,
        label_column=args.label_column,
        classes=classes,
    )


if __name__ == "__main__":
    finetune_esm2_entrypoint()
    flare.shutdown()
