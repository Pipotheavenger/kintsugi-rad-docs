"""Command-line entry point: `python scripts/launcher.py <mode> config.yaml`.

Stage: launch. Builds LauncherConfig (model + data) from the YAML, sets up the
Lightning Trainer, W&B logging and callbacks, then trains, evaluates, caches
backbone outputs or exports the model (.pt2).
"""

import argparse
import os
import shutil
import tempfile
import time
import typing
from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from kirad.datasets_and_dataloaders import UuidSampler
from kirad.launcher_utils import (
    DistributedModelCheckpoint,
    LauncherConfig,
    LauncherLogger,
)
from kirad.utils import cache_backbone, get_package_info, get_package_path

LauncherParams = namedtuple("LauncherParams", ["launcher_config", "launcher_logger"])
Modes = Literal["train", "evaluate", "cache", "compile"]


def launcher(
    mode: Modes,
    config: Mapping | str | Path,
    experiment_dir: Optional[str | Path] = None,
):
    """Run one mode (train/evaluate/cache/compile) end to end from a YAML config.

    Builds config, callbacks and Trainer (GPU fp16 or CPU fp32, DDP), then dispatches.
    Note: training_examples_per_eval must be a multiple of the global batch size.
    """
    if mode not in typing.get_args(Modes):
        raise ValueError(
            f"Unrecognized mode {mode}. Valid modes are: {typing.get_args(Modes)}."
        )
    params = _setup_launcher(config, experiment_dir)
    callbacks = _setup_callbacks(params.launcher_config)
    _log_to_wandb(*params)

    num_gpus = params.launcher_config.num_gpus
    training_params = params.launcher_config.training_params
    training_examples_per_eval = training_params.training_examples_per_eval

    if training_examples_per_eval is None:
        val_args = dict(check_val_every_n_epoch=1, val_check_interval=None)
    elif training_examples_per_eval % training_params._global_batch_size != 0:
        raise ValueError(
            f"Invalid training_examples_per_eval: {training_examples_per_eval}"
            f" is not divisible by global_batch_size = "
            f"{training_params._global_batch_size}"
        )
    else:
        val_args = dict(
            check_val_every_n_epoch=None,
            val_check_interval=training_examples_per_eval
            // training_params._global_batch_size,
        )

    trainer = Trainer(
        accelerator="gpu" if num_gpus > 0 else "cpu",
        num_nodes=int(os.environ.get("NUM_NODES", 1)),
        # HARDCODED: DDP with find_unused_parameters=True; fp16 on GPU, fp32 on CPU.
        strategy=DDPStrategy(
            find_unused_parameters=True
        ),  # LoRA seems to need this set to True
        precision=16 if num_gpus > 0 else 32,
        logger=params.launcher_logger.wandb_logger
        if params.launcher_logger is not None
        else None,
        accumulate_grad_batches=training_params.total_accum_grad_batches,
        max_epochs=params.launcher_config.training_params.max_epochs,
        log_every_n_steps=5,
        devices=num_gpus if num_gpus > 0 else "auto",
        callbacks=callbacks,
        **val_args,
    )

    if mode == "train":
        run_training(trainer, params.launcher_config)
    elif mode == "evaluate":
        run_evaluation(trainer, params.launcher_config)
    elif mode == "cache":
        run_caching(trainer, params.launcher_config, params.launcher_logger)
    elif mode == "compile":
        run_compilation(params.launcher_config)

    if mode != "compile":
        _log_datasets_to_wandb(*params)


def run_training(trainer: Trainer, launcher_config: LauncherConfig):
    """Fit on train/val, reload the best checkpoint, then test it on the test split."""
    train_loader = launcher_config.get_data_loader("train", training_mode=True)
    val_loader = launcher_config.get_data_loader("val", training_mode=False)
    test_loader = launcher_config.get_data_loader("test", training_mode=False)
    trainer.fit(
        launcher_config.model,
        train_loader,
        val_loader,
        ckpt_path=launcher_config.restore_training_state_ckpt_path,
    )

    #
    launcher_config.model.load_state_dict(
        torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
    )
    trainer.test(launcher_config.model, test_loader)


def run_evaluation(trainer: Trainer, launcher_config: LauncherConfig):
    """Test a trained checkpoint (model_params.ckpt_path) on the test split."""
    if launcher_config.model_params.ckpt_path == "":
        raise ValueError("Need to set a checkpoint path (`ckpt_path`) for evaluation.")

    test_loader = launcher_config.get_data_loader("test", training_mode=False)
    trainer.test(
        launcher_config.model,
        test_loader,
        ckpt_path=launcher_config.restore_training_state_ckpt_path,
    )


def _setup_launcher(
    config: Mapping | str | Path, experiment_dir: Optional[str | Path] = None
) -> LauncherParams:
    """Load the YAML into LauncherConfig and create the W&B logger on rank 0 only.

    Note: wandb_entity/wandb_project are hardcoded to None and raise; set them here.
    """

    # some things need to be done only for one process
    is_init_logger = (
        int(os.environ.get("NODE_RANK", 0)) == 0
        and int(os.environ.get("LOCAL_RANK", 0)) == 0
    )

    launcher_config = LauncherConfig(config, experiment_dir)
    # HARDCODED: W&B entity/project are None, so every run raises until you set them.
    wandb_entity = None
    wandb_project = None
    if wandb_entity is None:
        raise ValueError("Need to define `wandb_entity`.")
    if wandb_project is None:
        raise ValueError("Need to define `wandb_project`.")
    launcher_logger = (
        LauncherLogger("./logs", launcher_config.config, wandb_entity, wandb_project)
        if is_init_logger
        else None
    )

    return LauncherParams(
        launcher_config=launcher_config,
        launcher_logger=launcher_logger,
    )


def _setup_callbacks(launcher_config: LauncherConfig) -> list[pl.callbacks.Callback]:
    """Best-checkpoint saver, early stopping and LR monitor on "mode:metric" monitor_loss.

    Hardcoded: keeps top-1 + last.ckpt in chkpts/<experiment_name>/ (/mnt/chkpts on k8s).
    """
    (
        monitor_loss_mode,
        monitor_loss_name,
    ) = launcher_config.training_params.monitor_loss.split(":")
    # on Kubernetes space on OS disk can be low so store checkpoints to the external disk
    # HARDCODED: checkpoint root; /mnt/chkpts when JOB_NAME (Kubernetes) is set.
    chkpt_dir = "/mnt/chkpts" if "JOB_NAME" in os.environ else "chkpts"
    checkpoint_callback = DistributedModelCheckpoint(
        dirpath=f"{chkpt_dir}/{launcher_config.experiment_name}/",
        filename="epoch-{epoch}_metric-{monitor_loss_name:.3f}",
        monitor=monitor_loss_name,
        save_last=True,
        save_top_k=1,
        mode=monitor_loss_mode,
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_on_train_epoch_end=False,  # Save at end of val to properly checkpoint val-tuned thresholds
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stopping = EarlyStopping(
        monitor=monitor_loss_name,
        mode=monitor_loss_mode,
        patience=launcher_config.training_params.early_stopping_patience,
        strict=not launcher_config.model_params.restore_training_state,
    )

    callbacks = [lr_monitor, early_stopping, checkpoint_callback]

    return callbacks


def _log_to_wandb(launcher_config: LauncherConfig, launcher_logger: LauncherLogger):
    """Upload experiment + kirad source, package versions and ideal log-mel energies to W&B.

    Note: get_package_info("kipy") fails without the private kipy package.
    """

    if launcher_logger is not None:
        # Log the code
        launcher_logger.log_code(launcher_config.experiment_dir, "source-experiment")
        try:
            kintsugi_rad_path = get_package_path("kirad")
            launcher_logger.log_code(kintsugi_rad_path / "kirad", "source-kirad")
        except:
            logger.warning(
                "Unable to resolve the path to kintsugi-rad. Skip logging the `kirad` "
                "source code. Set KINTSUGI_RAD_PATH in the environment to specify the "
                "path to kintsugi-rad."
            )

        # Log kipy and kirad info
        data = []
        # HARDCODED: "kipy" is a private Kintsugi package; remove it if unavailable.
        for package in ["kipy", "kirad"]:
            package_info = get_package_info(package)
            data.append(package_info)
        package_info_df = pd.DataFrame(data)
        launcher_logger.log_table_artifacts(
            {"package_info": {"package_info": package_info_df}}
        )

        # Log ideal logmel energies, if specified
        if "audio" in launcher_config.modality_params:
            modality_params = launcher_config.modality_params["audio"]
            for split in ["train", "val", "test"]:
                if (
                    split in modality_params
                    and modality_params[split].get("ideal_logmel_energies") is not None
                ):
                    # Assume that the same ideal logmel energies are used for each
                    # split, so only need to log once.
                    ideal_logmel_energies = np.load(
                        modality_params[split]["ideal_logmel_energies"]
                    )
                    df = pd.DataFrame(
                        columns=["ideal_logmel_energies"], data=ideal_logmel_energies
                    )
                    launcher_logger.log_table_artifacts(
                        {"ideal_logmel_energies": {"ideal_logmel_energies.npz": df}}
                    )
                    break


def _log_datasets_to_wandb(
    launcher_config: LauncherConfig, launcher_logger: LauncherLogger
):
    """Upload the metadata table actually used for each split as a W&B "dataset" artifact."""
    if launcher_logger is not None:
        # Log dataset splits that were used
        used_datasets = dict()
        for split, dataset in launcher_config.dataset.items():
            used_datasets.update({f"{split}_dataset": dataset.datasets["metadata"]._df})
        launcher_logger.log_table_artifacts({"dataset": used_datasets})


def run_caching(
    trainer: Trainer, launcher_config: LauncherConfig, launcher_logger: LauncherLogger
):
    """Run the backbone over train/val/test and store its outputs in W&B (cache mode).

    Requires GPUs and no backbone_cache in data_params; a dummy predict() starts DDP.
    """
    # prohibit running caching if cache is set
    assert "backbone_cache" not in launcher_config.config["data_params"]

    datasets, data_loader_args = [], []
    for split in ["train", "val", "test"]:
        datasets.append(launcher_config.get_dataset(split))
        data_loader_args.append(launcher_config.get_data_loader_kwargs(split, False))

    # Warning: before running caching procedure, the distributed environment
    # should be properly initialized. It happens under the hood when we call
    # trainer.fit(), .test() or .predict(), so one of these methods should be called
    # prior to caching. For this reason we call .predict() with 1 sample per gpus.
    dummy_loader = torch.utils.data.DataLoader(datasets[-1], **data_loader_args[-1])
    samples = [next(iter(dummy_loader))] * torch.cuda.device_count()
    sample_loader = torch.utils.data.DataLoader(
        samples, batch_size=1, collate_fn=lambda x: x[0]
    )
    trainer.predict(
        launcher_config.model,
        sample_loader,
        ckpt_path=launcher_config.restore_training_state_ckpt_path,
    )

    assert torch.distributed.is_initialized()

    # replace sampler with the distributed one
    for dataset, arg in zip(datasets, data_loader_args):
        arg["sampler"] = UuidSampler(dataset.datasets["metadata"], distributed=True)

    #
    launcher_config.model.cuda(torch.distributed.get_rank()).eval()

    state_dict = launcher_config.model.backbone.state_dict(prefix="backbone.")
    cache_backbone(
        launcher_config.model.compute_features_to_cache,
        launcher_logger.wandb_logger if launcher_logger else None,
        datasets,
        data_loader_args,
        state_dict,
    )


def run_compilation(launcher_config: LauncherConfig):
    """Export the model with torch.export on one test window, check outputs, log .pt2 to W&B.

    Note: uses only the audio feature; needs an active wandb.run.
    """

    # we need exactly one sample for running a compiler
    dummy_dataset = launcher_config.get_dataset("test")
    dummy_data_loader = launcher_config.get_data_loader_kwargs("test", False)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, **dummy_data_loader)
    batch = next(iter(dummy_loader))
    sample_input = ({"audio": batch["features"]["audio"][:1, ...]}, [1])

    #
    launcher_config.model.eval()
    model_compiled = torch.export.export(launcher_config.model, sample_input)

    # sanity check
    output_original, _ = launcher_config.model(*sample_input)
    output_compiled, _ = model_compiled.module()(*sample_input)
    for key in output_compiled:
        if not torch.allclose(output_compiled[key], output_original[key]):
            raise ValueError("Computation results don't match")

    #
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "model.pt2")

    torch.export.save(model_compiled, path)

    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
    artifact.add_file(path)
    wandb.log_artifact(artifact)

    shutil.rmtree(temp_dir, ignore_errors=True)


def _parse_args():
    """Parse CLI args: mode, config path, optional --experiment-dir."""
    parser = argparse.ArgumentParser(
        description="Launcher for training, tuning, and evaluation."
    )

    # Positional arguments
    parser.add_argument(
        "mode",
        type=str,
        choices=typing.get_args(Modes),
        help="Choose whether to operate in training, tuning, or evaluation mode.",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to a YAML config that defines the network and training parameters.",
    )

    # Optional arguments
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Path to the experiment directory that contains experiment artifacts, "
        "such as model.py and ideal_logmel_energies.npy. If not specified, defaults "
        "to the parent directory of the config file.",
    )

    args = parser.parse_args()

    return args


def setup_multi_node_environment():
    """Wait for all Kubernetes job pods, then set MASTER_ADDR/PORT, WORLD_SIZE, NODE_RANK."""

    # this is a bit hacky.
    # kubernetes package is only required inside docker container worker,
    # not necessarily in the kipy
    from kubernetes import client, config

    #
    config.load_incluster_config()
    core_v1 = client.CoreV1Api()

    #
    job_name = os.environ["JOB_NAME"]
    num_nodes = int(os.environ.get("NUM_NODES", 1))

    #
    start_time = time.time()
    while True:

        # get pods associated with this job
        pods = core_v1.list_namespaced_pod(
            # HARDCODED: Kubernetes namespace "default".
            namespace="default", label_selector=f"job-name={job_name}"
        )

        active_nodes = sum(pod.status.pod_ip is not None for pod in pods.items)
        if active_nodes == num_nodes:
            break

        print(f"{active_nodes}/{num_nodes} nodes are ready. Waiting ...", flush=True)

        # we give it at most 20 mins to warmup.
        # In most cases it should be less than few mins.
        if time.time() - start_time > 20 * 60:
            print(
                f"Max wait time has been exceeded. Target num nodes {num_nodes} has not been reached. Shutting down."
            )
            exit(1)

        time.sleep(10)

    # setup env variables for multi-node training
    # node with JOB_COMPLETION_INDEX == 0 is a master node
    os.environ["MASTER_ADDR"] = [
        pod.status.pod_ip
        for pod in pods.items
        if pod.metadata.annotations["batch.kubernetes.io/job-completion-index"] == "0"
    ][0]
    # HARDCODED: DDP master port 12355.
    os.environ["MASTER_PORT"] = "12355"  # some arbitrary recommended port
    os.environ["WORLD_SIZE"] = str(int(num_nodes * torch.cuda.device_count()))
    os.environ["NODE_RANK"] = os.environ["JOB_COMPLETION_INDEX"]


if __name__ == "__main__":
    args = _parse_args()

    # we are in the Kubernetes mode
    if "JOB_NAME" in os.environ:
        setup_multi_node_environment()

    launcher(args.mode, args.config, experiment_dir=args.experiment_dir)
