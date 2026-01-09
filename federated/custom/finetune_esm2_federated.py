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

"""
Repo-style NVFlare adaptation of BioNeMo ESM2 finetuning for federated learning.

Key goals:
- Use nvflare.client.lightning.patch(trainer, ...) so FL state transfer is handled correctly.
- Receive FLModel once per round for round/site metadata (not for manual weight loading).
- Robust TensorBoard logging:
    * Create exactly ONE TB logger per site per round
    * Attach it to the Lightning Trainer (trainer.logger)
    * Avoid double-logging via NeMo logger
- Avoid local checkpoint collisions (common error: "checkpoint destination directory is not empty"):
    * Disable per-client checkpoint callbacks by default in FL
    * Server-side aggregation persists the global model; client local checkpoints are optional
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.dataset import (
    InMemoryProteinDataset,
    InMemorySingleValueDataset,
)
from bionemo.esm2.model.finetune.sequence_model import ESM2FineTuneSeqConfig

from bionemo.esm2.scripts.finetune_esm2 import get_parser
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.model.config import TorchmetricsConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger

from lightning.pytorch.callbacks import Callback, LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule

# NVFlare Lightning client API (repo style)
import nvflare.client.lightning as flare


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return default


def train_model(
    train_data_path: Path,
    valid_data_path: Path,
    num_nodes: int,
    devices: int,
    min_seq_length: Optional[int],
    max_seq_length: int,
    result_dir: Path,
    num_steps: int,
    limit_val_batches: float,
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
    # single value classification / regression mlp
    mlp_ft_dropout: float = 0.25,
    mlp_hidden_size: int = 256,
    mlp_target_size: int = 1,
    # token-level classification cnn
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
    classes: list[str] | None = None,
) -> tuple[Optional[Path], Callback | None, nl.Trainer]:
    """
    Federated client training entrypoint. This runs ONE local training "round".

    Important:
    - In FL, the server usually stores the aggregated global model.
    - Client local checkpoints are typically disabled to avoid directory collisions and disk bloat.
    """

    result_dir.mkdir(parents=True, exist_ok=True)

    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

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

    callbacks: list[Callback] = [
    RichModelSummary(max_depth=4),
    nl_callbacks.PreemptionCallback(),
    ]
    if create_tensorboard_logger:
        callbacks.insert(1, LearningRateMonitor())
    if metric_tracker is not None:
        callbacks.append(metric_tracker)

    if nsys_profiling:
        if nsys_end_step is None:
            nsys_end_step = num_steps
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=nsys_start_step,
                end_step=nsys_end_step,
                ranks=nsys_ranks,
                gen_shape=True,
            )
        )
    

    

    # ------------------------------------------------------------
    # 1) Create Trainer first (no TB logger yet), then patch, then receive.
    #    We set TB logger AFTER we know site/round to avoid collisions.
    # ------------------------------------------------------------
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
        logger=False,  # keep False here intentionally
        enable_checkpointing=False,
        plugins=nl.MegatronMixedPrecision(
            precision=precision,
            params_dtype=get_autocast_dtype(precision),
            pipeline_dtype=get_autocast_dtype(precision),
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            autocast_enabled=False,
        ),
    )

    # --- CRITICAL: repo-style patch ---
    # This wires up FL state handling into Trainer fit loop.
    flare.patch(trainer, restore_state=False, load_state_dict_strict=False)

    # Receive FLModel (use it for round/site metadata; patch handles weights internally)
    input_model = flare.receive()
    current_round = int(getattr(input_model, "current_round", 0))
    site = flare.get_site_name()

    print(
        f"\n[FL] Site={site} Round={current_round} "
        f"(params={len(input_model.params) if getattr(input_model, 'params', None) else 0})\n"
    )

    # ------------------------------------------------------------
    # 2) Round-scoped output directory (important for TB + avoiding collisions)
    # ------------------------------------------------------------
    round_dir = result_dir / f"round{current_round}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 3) TensorBoard (ONE logger per site per round)
    # ------------------------------------------------------------
    if create_tensorboard_logger:
        tb_logger = TensorBoardLogger(
            save_dir=str(round_dir / "tb_logs"),
            name=experiment_name,
            version=f"{site}_round{current_round}",
        )
        trainer.logger = tb_logger


    # Optional: add your custom TB streamer if it exists (it can push round-aligned steps)
    try:
        from custom.bionemo_tb_streamer import BioNeMoTBStreamer  # type: ignore
        start_step = current_round * num_steps
        trainer.callbacks.append(BioNeMoTBStreamer(start_step=start_step))
        print(f"[TB] Attached BioNeMoTBStreamer(start_step={start_step})")
    except Exception:
        # Not required; TB will still work via trainer.logger if enabled
        pass

    # Optional: clean previous round local checkpoints if your environment creates them anyway
    # (Some stacks may still create dist-ckpt folders during training)
    keep_last_ckpt_only = True
    if keep_last_ckpt_only and current_round > 0:
        prev_round_dir = result_dir / f"round{current_round - 1}"
        prev_ckpt_dir = prev_round_dir / experiment_name / "dev" / "checkpoints"
        if prev_ckpt_dir.is_dir():
            try:
                print(f"[Cleanup] Removing previous checkpoint directory: {prev_ckpt_dir}")
                shutil.rmtree(prev_ckpt_dir)
            except Exception as e:
                print(f"[Cleanup] Warning: failed to remove {prev_ckpt_dir}: {repr(e)}")

    # ------------------------------------------------------------
    # 4) LR schedule tweak per round (optional)
    # ------------------------------------------------------------
    if current_round > 0:
        lr_step_reduce = 1.05
        denom = max(1.0, current_round * lr_step_reduce)
        new_lr = lr / denom
        new_lr_multiplier = lr_multiplier / denom
        print(f"[LR] Round={current_round}: lr {lr} -> {new_lr} (denom={denom})")
    else:
        new_lr = lr
        new_lr_multiplier = lr_multiplier

    # ------------------------------------------------------------
    # 5) Data
    # ------------------------------------------------------------
    tokenizer = get_tokenizer()

    train_dataset = dataset_class.from_csv(
        train_data_path, task_type=task_type, label_column=label_column
    )
    valid_dataset = dataset_class.from_csv(
        valid_data_path, task_type=task_type, label_column=label_column
    )

    if task_type == "classification":
        if classes:
            if not isinstance(classes, list):
                raise ValueError(f"classes must be list[str], got {type(classes)}: {classes}")
            train_dataset.label_tokenizer.build_vocab([classes])
            print(f"[Data] Built label vocab from classes: {classes}")

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

    # ------------------------------------------------------------
    # 6) Metrics config
    # ------------------------------------------------------------
    train_metric = None
    if task_type == "regression":
        valid_metric = TorchmetricsConfig(
            class_path="MeanSquaredError",
            task="regression",
            metric_name="val_mse",
        )
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

    # ------------------------------------------------------------
    # 7) Model config (ESM2 finetune)
    #    NOTE: This DOES add/enable the task head (regression head for task_type=regression).
    # ------------------------------------------------------------
    config = config_class(
        task_type=task_type,
        encoder_frozen=encoder_frozen,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=str(restore_from_checkpoint_path) if restore_from_checkpoint_path else None,
        # In FL: skip loading head weights from the base checkpoint so head can be trained cleanly
        initial_ckpt_skip_keys_with_these_prefixes=[f"{task_type}_head"],
        train_metric=train_metric,
        valid_metric=valid_metric,
    )

    # Apply task-dependent head params if present on config
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
        )
    )
    if scale_lr_layer:
        optimizer.scale_lr_cond = lambda name, param: scale_lr_layer in name
        optimizer.lr_mult = new_lr_multiplier

    module = biobert_lightning_module(config=config, tokenizer=tokenizer, optimizer=optimizer)

    # ------------------------------------------------------------
    # 8) NeMo logger (avoid double TB)
    #    We rely on trainer.logger for TensorBoard if enabled.
    # ------------------------------------------------------------
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=round_dir,
        name=experiment_name,
        initialize_tensorboard_logger=False,  # IMPORTANT: avoid duplicating TB logs
        wandb_config=wandb_config,
        ckpt_callback=None,  # FL: no per-client checkpoint callback by default
    )

    # ------------------------------------------------------------
    # 9) Train
    # ------------------------------------------------------------
    llm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=None,
    )


    # Helpful debug: show which metrics Lightning knows about
    try:
        print("[Metrics] callback_metrics keys:", list(trainer.callback_metrics.keys()))
        print("[Metrics] logged_metrics keys:", list(trainer.logged_metrics.keys()))
    except Exception:
        pass

    # In FL, we generally do NOT return a checkpoint path (server aggregates model state)
    ckpt_path = None
    return ckpt_path, metric_tracker, trainer


def finetune_esm2_entrypoint() -> None:
    flare.init()

    parser = get_parser()

    # Add a small convenience option for classification class labels
    parser.add_argument(
        "--classes",
        type=str,
        required=False,
        default=None,
        help="Comma-separated class names for classification, e.g. 'pos,neg'. Only used when --task-type classification.",
    )

    args = parser.parse_args()

    if args.classes:
        if args.task_type != "classification":
            parser.error("Use --classes only with --task-type 'classification'")
        classes = args.classes.split(",")
    else:
        classes = None

    # Convert dataset_class string -> class object (needed for CLI usage)
    if isinstance(args.dataset_class, str):
        if args.dataset_class == "InMemorySingleValueDataset":
            actual_dataset_class = InMemorySingleValueDataset
        elif args.dataset_class == "InMemoryProteinDataset":
            actual_dataset_class = InMemoryProteinDataset
        else:
            print(
                f"Warning: Unknown dataset_class string '{args.dataset_class}'. "
                "Defaulting to InMemorySingleValueDataset."
            )
            actual_dataset_class = InMemorySingleValueDataset
    else:
        actual_dataset_class = args.dataset_class

    # Avoid padding for single value labels:
    if args.min_seq_length is not None and actual_dataset_class is InMemorySingleValueDataset:
        parser.error("Arguments --min-seq-length cannot be set when using InMemorySingleValueDataset.")

    # Note: args.create_tensorboard_logger is expected from BioNeMo parser
    train_model(
        train_data_path=Path(args.train_data_path),
        valid_data_path=Path(args.valid_data_path),
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        result_dir=Path(args.result_dir),
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        wandb_group=args.wandb_group,
        wandb_id=args.wandb_id,
        wandb_anonymous=args.wandb_anonymous,
        wandb_log_model=args.wandb_log_model,
        wandb_offline=args.wandb_offline,
        num_steps=args.num_steps,
        limit_val_batches=_safe_float(args.limit_val_batches, default=1.0),
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        num_dataset_workers=args.num_dataset_workers,
        lr=float(args.lr),
        micro_batch_size=args.micro_batch_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        task_type=args.task_type,
        encoder_frozen=args.encoder_frozen,
        scale_lr_layer=args.scale_lr_layer,
        lr_multiplier=args.lr_multiplier,
        mlp_ft_dropout=args.mlp_ft_dropout,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_target_size=args.mlp_target_size,
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
        dataset_class=actual_dataset_class,
        config_class=args.config_class,
        overlap_grad_reduce=args.overlap_grad_reduce,
        overlap_param_gather=not args.no_overlap_param_gather,
        average_in_collective=not args.no_average_in_collective,
        grad_reduce_in_fp32=args.grad_reduce_in_fp32,
        label_column=args.label_column,
        classes=classes,
        create_tensorboard_logger=bool(getattr(args, "create_tensorboard_logger", False)),
    )


if __name__ == "__main__":
    finetune_esm2_entrypoint()
    flare.shutdown()
