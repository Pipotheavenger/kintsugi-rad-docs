"""Turns the YAML config into objects: params, model, datasets, loaders, W&B logger.

Stage: launch (plus outputs: checkpoint replication and W&B artifact logging).
Used by scripts/launcher.py.
"""

import copy
import os
import socket
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, StackDataset

from .constants import CPU_DEVICE, DatasetSplit
from .dataset_utils import broadcast_collate
from .datasets_and_dataloaders import KintsugiMetadata, UuidSampler
from .utils import (
    dataframe_to_wandb_table,
    get_wandb_run_name_tag_from_artifact_url,
    import_object,
    localize_wandb_artifact,
)

SPLITS = ["train", "val", "test"]


# The standard ModelCheckpoint allows checkpoint saving only on the main node,
# but this class extends its functionality to store "best" checkpoint replicas on
# all nodes in the cluster.
class DistributedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that also keeps a copy of the best checkpoint on every node."""

    def on_validation_end(self, trainer, pl_module):
        """Save best/last ckpt after validation (with tuned thresholds); replicate on nodes.

        Non-zero nodes (local rank 0) store only {"state_dict"} and delete the old best.
        Note: needs torch.distributed initialized (uses barriers).
        """
        best_ckpt_prev = self.best_model_path
        torch.distributed.barrier()

        super().on_validation_end(trainer, pl_module)
        torch.distributed.barrier()

        best_ckpt_curr = self.best_model_path

        # Save checkpoints only for local_rank == 0 on each node, except the first node
        # because saving there is already handled by on_validation_end()
        local_rank = torch.distributed.get_node_local_rank(fallback_rank=0)
        global_rank = torch.distributed.get_rank()
        if global_rank != 0 and local_rank == 0:
            # checkpoints can be heavy so remove old best checkpoint to save space
            # if the best checkpoint has changed, remove the old one
            if best_ckpt_prev != best_ckpt_curr and os.path.exists(best_ckpt_prev):
                os.remove(best_ckpt_prev)

            # save checkpoint if it doesn't exist already
            if best_ckpt_curr and not os.path.exists(best_ckpt_curr):
                os.makedirs(os.path.dirname(best_ckpt_curr), exist_ok=True)
                torch.save(
                    {"state_dict": trainer.lightning_module.state_dict()},
                    best_ckpt_curr,
                )


class SchedulerParams(BaseModel):
    """LR scheduler settings from training_params.scheduler (gamma, milestones)."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    gamma: float = 0.0
    milestones: list[int] = []


class OptimizerParams(BaseModel):
    """Optimizer settings for one parameter group (lr, weight_decay)."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    lr: float
    weight_decay: float = 0.0


class TrainingParams(BaseModel):
    """Validated training_params: batch sizes, epochs, monitor metric, optimizer, seed.

    Grad accumulation = effective_batch_size / (world_size * batch_size); must divide.
    Note: constructing it calls pl.seed_everything(seed).
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    batch_size: int
    effective_batch_size: int
    max_epochs: int
    monitor_loss: str
    scheduler: SchedulerParams
    optimizer: Mapping[str, OptimizerParams]
    tasks: Optional[Mapping[str, Any]] = None
    seed: int = 42  # HARDCODED: default seed
    early_stopping_patience: int = 10  # HARDCODED: default patience (validations)
    training_examples_per_eval: Optional[int] = None  # None -> once per epoch
    deterministic: bool = True
    cpus_per_worker: Optional[int] = None  # None -> (num_cpus - 1) // num_gpus

    _global_batch_size: int = PrivateAttr()

    def __init__(self, *args, **kwargs):
        """Compute the global batch size (WORLD_SIZE x batch_size) and seed everything."""
        super().__init__(*args, **kwargs)
        num_gpus = torch.cuda.device_count()
        world_size = int(os.environ.get("WORLD_SIZE", num_gpus)) if num_gpus > 0 else 1
        self._global_batch_size = world_size * self.batch_size
        # Set seed as per config and pass True flag to workers so that they get same
        # seed in their processes.
        pl.seed_everything(self.seed, workers=True)

    @property
    def total_accum_grad_batches(self):
        """Gradient-accumulation steps: effective_batch_size // global batch size."""
        quot, rem = divmod(self.effective_batch_size, self._global_batch_size)
        if rem != 0:
            raise ValueError(
                f"Invalid effective batch size: {self.effective_batch_size} "
                f"is not divisible by global_batch_size = "
                f"{self._global_batch_size}."
            )
        return quot


class ModelParams(BaseModel):
    """Validated model_params: model class path, config kwargs and checkpoint to load.

    ckpt_path may be local, a W&B artifact URL or "entity/project/run:tag"; W&B ones
    are downloaded. A data_params backbone_cache replaces ckpt_path (not both).
    Note: restore_training_state with a local ckpt_path fails (`run` is undefined).
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    model_path: str
    ckpt_path: Optional[str | Path] = None
    config: Mapping[str, Any] = dict()
    restore_training_state: bool = False

    _cached_backbone: bool = PrivateAttr()

    def __init__(self, backbone_cache: Optional[str] = None, *args, **kwargs):
        """Resolve ckpt_path (W&B download or backbone cache) and check restored config."""
        super().__init__(*args, **kwargs)

        self._cached_backbone = False
        if self.ckpt_path and backbone_cache:
            # Setting ckpt_path and backbone_cache is not allowed because one can
            # inadvertently set mismatching paths (eg. ckpt_path contains model weights
            # for backbone #1 while backbone_cache contains cached outputs from
            # backbone #2).
            raise ValueError(
                "Cannot set both `ckpt_path` under 'model_params' and `backbone_cache` "
                "under 'data_params' at the same time."
            )
        elif backbone_cache is not None and backbone_cache != "":
            self.ckpt_path = backbone_cache
            self._cached_backbone = True

        # If we don't restore training state we will continue training from scratch
        # (epoch #0) and potentially with new optimizer params, but with weights
        # initialized from the checkpoint.
        if self.ckpt_path is not None and self.ckpt_path != "":
            if self.ckpt_path.startswith("https://"):
                run, name, tag = get_wandb_run_name_tag_from_artifact_url(
                    self.ckpt_path
                )
                self.ckpt_path = localize_wandb_artifact(run, name, tag)
            elif ":" in self.ckpt_path:
                # in case the model checkpoint is provided in the format run_id:tag
                # rather than full wandb path
                run_id, tag = self.ckpt_path.split(":")
                run_id_parts = run_id.split(
                    "/"
                )  # should be of form "entity/project/run"
                name = f"model-{run_id_parts[-1]}"
                run = wandb.Api().run("/".join(run_id_parts))
                self.ckpt_path = localize_wandb_artifact(run, name, tag)

            # Use model params fetched from wandb only when we are resuming full
            # training state because otherwise we most likely want to modify some of the
            # training params in which case the local config is preferred
            if self.restore_training_state:
                if self.config != run.config["model_params"]["config"]:
                    current_config_yaml_str = yaml.safe_dump(self.config)
                    restored_config_yaml_str = yaml.safe_dump(
                        run.config["model_params"]["config"]
                    )
                    raise ValueError(
                        "The model parameters specified in the current config don't "
                        "match the parameters specified in the config for the run "
                        "that you are restoring. Check the config in "
                        f"{'/'.join(run.path)}."
                        f"\n\nCurrent config:\n{current_config_yaml_str}"
                        f"\n\nRestored run config:\n{restored_config_yaml_str}"
                    )
        else:
            self.ckpt_path = None


class LauncherConfig:
    """Everything built from one YAML config; datasets and loaders are created lazily."""

    def __init__(
        self,
        config: str | Path | Mapping,
        experiment_dir: str | Path | None = None,
    ):
        """Read YAML, parse training/data/model params and instantiate the model.

        experiment_dir (default: the config's folder) is added to sys.path so the
        experiment's model.py can be imported by model_path.
        """
        if experiment_dir is None:
            if isinstance(config, Mapping):
                raise ValueError(
                    "Must set experiment directory if setting the config with a dictionary."
                )
            # Set experiment directory to the directory containing the config.
            self.experiment_dir = Path(config).expanduser().resolve().parent
        else:
            self.experiment_dir = Path(experiment_dir).resolve()
        if isinstance(config, str | Path):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        self.config = config

        self.experiment_name = config["experiment_name"]
        self.description = config["description"]
        sys.path.append(str(self.experiment_dir))

        if "training_params" in config and len(config["training_params"]) > 0:
            self.training_params = TrainingParams(**config["training_params"])
        else:
            logger.info("No training params specified. Only suitable for inference.")
            self.training_params = None
        self._init_compute_params()
        self._init_data_params()
        # Check if backbone cache is specified in order to resolve the behavior of
        # `ckpt_path` in "model_params".
        if "backbone_cache" in self.modality_params:
            # We've been using the same backbone cache for all data splits (the
            # Datasets for each split take care of subsetting into train/val/test), so
            # arbitrarily get the backbone cache from the "train" split.
            backbone_cache = self.modality_params["backbone_cache"]["train"][
                "backbone_cache"
            ]
        else:
            backbone_cache = None
        self.model_params = ModelParams(
            **config["model_params"], backbone_cache=backbone_cache
        )
        self._init_model()

        # Store the datasets and associated data loaders. Dictionary is a mapping from
        # dataset split to the corresponding dataset/data loader. We will populate the
        # datasets and data loaders only when requested.
        self.metadata = dict()
        self.dataset = dict()
        self.data_loader = dict()

    def _init_compute_params(self):
        """Count GPUs/CPUs, set DataLoader workers per GPU and deterministic torch/CuBLAS."""
        # Add GCE instance name to the config
        try:
            self.config["gce_instance_name"] = socket.gethostname()
        except:
            pass

        # Run 1 GPU per worker for training. If there are no GPUs, then use just 1
        # worker (all CPUs will be assigned to the 1 worker). 1 CPU is allocated to the
        # trainer (by default), so assign one less CPU per worker.
        self.num_gpus = torch.cuda.device_count()
        self.num_cpus = os.cpu_count()
        if (
            self.training_params is not None
            and self.training_params.cpus_per_worker is not None
        ):
            self.cpus_per_worker = self.training_params.cpus_per_worker
        else:
            self.cpus_per_worker = (self.num_cpus - 1) // max(self.num_gpus, 1)

        # Option for deterministic algorithms:
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        # torch.use_deterministic_algorithms(mode=True, warn_only=True)
        torch.use_deterministic_algorithms(
            mode=self.training_params is None or self.training_params.deterministic,
            warn_only=True,
        )

        # Makes certain CuBLAS operations deterministic:
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        # HARDCODED: CuBLAS workspace setting required for deterministic GPU ops.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def _init_data_params(self):
        """Parse data_params keys "<modality>=<class.path>" into factories + per-split params.

        "metadata=..." is required; every other key (audio, text, backbone_cache) is a
        modality dataset. Params per split come from get_params_by_split.
        """
        self.metadata_factory = None
        self.modality_to_dataset_factory = dict()
        self.modality_params = dict()
        for modality_and_class_path, class_params in self.config["data_params"].items():
            modality, class_path = modality_and_class_path.split("=", maxsplit=1)
            if modality == "metadata":
                # Initialize the metadata parameters
                self.metadata_factory = import_object(class_path)
                self.metadata_params = dict()
                for split in SPLITS:
                    self.metadata_params[split] = self.get_params_by_split(
                        class_params, split, include_dataset_splits=True
                    )
            else:
                # Initialize dataset parameters for the modality (eg. audio, text)
                self.modality_to_dataset_factory[modality] = import_object(class_path)
                self.modality_params[modality] = dict()
                for split in SPLITS:
                    self.modality_params[modality][split] = self.get_params_by_split(
                        class_params, split, include_dataset_splits=False
                    )
        if self.metadata_factory is None:
            raise ValueError(
                "Missing metadata in the config. Must specify "
                "'metadata=<path.to.Metadata>' under 'data_params'."
            )
        if (
            "backbone_cache" in self.modality_to_dataset_factory.keys()
            and len(self.modality_to_dataset_factory) > 1
        ):
            # When using the backbone cache, keep in mind other possible modalities used (e.g. audio or
            # text). This happens in the LLMA training scenario.
            logger.warning("Backbone cache is used together with other modalities.")

    def _init_model(self):
        """Import model_path, build the model and load ckpt weights (or defer full restore).

        Weight loading falls back to strict=False on mismatch (partial init, printed).
        """
        # Set the model either from `kintsugi-rad` or `kipy`.
        model_factory = import_object(self.model_params.model_path)
        self.model = model_factory(
            **self.model_params.config,
            training_params=self.training_params,
            cached_backbone=self.model_params._cached_backbone,
        )
        if self.model_params.ckpt_path is not None:
            if self.model_params.restore_training_state:
                # Use trainer.fit(ckpt_path=<path_to_ckpt>) to restore the training state
                self.restore_training_state_ckpt_path = self.model_params.ckpt_path
            else:
                # Restore only the model weights from the checkpoint.
                # Setting to CPU saves GPU memory. It will be moved to GPU by Lightning
                # when needed.
                self.restore_training_state_ckpt_path = None
                state_dict = torch.load(
                    self.model_params.ckpt_path, map_location=CPU_DEVICE
                )["state_dict"]
                try:
                    self.model.load_state_dict(state_dict)
                except Exception as e:
                    print(
                        f"Performing partial initialization, some params are missing or extra (see below):\n {e}"
                    )
                    self.model.load_state_dict(state_dict, strict=False)
        else:
            # No checkpoint is specified
            self.restore_training_state_ckpt_path = None

    def get_metadata(self, split: DatasetSplit) -> KintsugiMetadata:
        """Build (once) and return the metadata object for a split."""
        try:
            metadata = self.metadata[split]
        except KeyError:
            metadata = self.metadata_factory(**self.metadata_params[split])
            self.metadata[split] = metadata

        return metadata

    def get_dataset(self, split: DatasetSplit) -> Dataset:
        """StackDataset of metadata + each modality dataset for a split (built once).

        Each item: {"metadata": ..., "audio": ..., "text": ...}, merged by broadcast_collate.
        """
        try:
            dataset = self.dataset[split]
        except KeyError:
            metadata = self.get_metadata(split)
            datasets = {"metadata": metadata}

            # Get datasets from requested modalities.
            for modality, dataset_factory in self.modality_to_dataset_factory.items():
                modality_params = self.modality_params[modality][split]
                modality_params["metadata"] = metadata
                datasets[modality] = dataset_factory(**modality_params)

            dataset = StackDataset(**datasets)
            self.dataset[split] = dataset

        return dataset

    def get_data_loader_kwargs(
        self, split: DatasetSplit, training_mode: bool
    ) -> Mapping:
        """DataLoader kwargs: broadcast_collate, UuidSampler (shuffled + drop_last if train)."""
        metadata = self.get_metadata(split)
        split_indep_kwargs = dict(
            batch_size=self.training_params.batch_size,
            num_workers=self.cpus_per_worker,
            collate_fn=broadcast_collate,
            persistent_workers=False,
        )
        if training_mode:
            split_kwargs = dict(
                drop_last=True, sampler=UuidSampler(metadata, shuffle=True)
            )
        else:
            # val and test
            split_kwargs = dict(
                drop_last=False, sampler=UuidSampler(metadata, shuffle=False)
            )

        return {**split_indep_kwargs, **split_kwargs}

    def get_data_loader(self, split: DatasetSplit, training_mode: bool) -> DataLoader:
        """DataLoader for a split (cached per split and mode); batch_size = recordings."""
        try:
            data_loader = self.data_loader[split, training_mode]
        except KeyError:
            dataset = self.get_dataset(split)

            data_loader_kwargs = self.get_data_loader_kwargs(split, training_mode)
            data_loader = DataLoader(dataset, **data_loader_kwargs)
            self.data_loader[split, training_mode] = data_loader

        return data_loader

    def resolve_artifact_path(self, artifact_path: str | Path) -> Path:
        """Resolve the absolute path for a file path specified in the config file.

        If the file path is absolute, then return the same path (after resolving
        symlinks). If the file path is relative, then assume that the path is relative
        to the experiment directory, so prepend the experiment directory to the
        file path.

        Parameters
        ----------
        artifact_path : str or Path
            Absolute or relative path to an artifact to be used by the trainer (eg.
            "ideal_logmel_energies.npz"). A relative path will be assumed to be
            relative to the experiment directory.

        Returns
        -------
        Path
            Absolute path (with symlinks resolved) to the artifact.
        """
        artifact_path = Path(artifact_path).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = self.experiment_dir / artifact_path
        return artifact_path.resolve()

    def resolve_artifact_paths_in_param_dict(
        self, param_dict: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Make relative string params absolute when they point to an existing file.

        Resolves the absolute paths for any file path parameters in the parameter
        dictionary.

        Parameters
        ----------
        param_dict : dict
            Mapping from parameter name to parameter value, which may or may not be a
            file path to some artifact.

        Returns
        -------
        dict
            Same format as the input parameter dictionary but with any artifact file
            paths resolved to be absolute.
        """
        for key, val in param_dict.items():
            if isinstance(val, str | Path):
                resolved_path = self.resolve_artifact_path(val)
                if resolved_path.exists():
                    # This is to ensure that we don't blindly treat all strings
                    # as file paths (eg. window_method = "all").
                    param_dict[key] = resolved_path

        return param_dict

    def get_params_by_split(
        self,
        all_splits_params: Mapping[str, Mapping[str, Any]],
        split: DatasetSplit,
        include_dataset_splits: bool,
    ) -> Mapping[str, Any]:
        """Merge config params for one split: default < val_test (val/test) < split.

        Return parameters for a given split after merging the parameters from
        "default" and "val_test" sections.

        Parameters for a given split are set in the following order of priority:
        1. parameters specified in the split-specific section
        2. parameters specified in the "val_test" section if the requested split is
            "val" or "test"
        3. parameters specified in the "default" section

        Parameters
        ----------
        all_splits_params : dict
            A dictionary that maps "train", "val", test", "val_test", and "default" to
            a dictionary of parameters.
        split : str
            Requested split to get parameters for. Can be "train", "val", or "test".
        include_dataset_splits : bool
            Indicate whether to include "dataset_splits" as a parameter. Needed when
            setting parameters for `KintsugiMetadata`.

        Returns
        -------
        dict
            A dictionary that maps parameter names to parameter values for the
            requested split after merging parameters from the "default" and "val_test"
            sections.
        """
        params = copy.deepcopy(all_splits_params.get("default", {}))
        if split in {"val", "test"}:
            params |= all_splits_params.get("val_test", {})
        params |= all_splits_params.get(split, {})
        if include_dataset_splits:
            params["dataset_splits"] = params.get("dataset_splits", split)
        params = self.resolve_artifact_paths_in_param_dict(params)

        return params


class LauncherLogger:
    """W&B run for the experiment; logs code and pandas tables as artifacts."""

    def __init__(
        self, log_path: str | Path, config: Mapping, entity: str, project: str
    ):
        """Start a WandbLogger (log_model=True uploads checkpoints) under log_path/wandb."""
        log_path = Path(log_path)
        wandb_log_path = log_path / "wandb"
        wandb_log_path.mkdir(parents=True, exist_ok=True)
        self.wandb_logger = WandbLogger(
            name=config["experiment_name"],
            save_dir=log_path,
            entity=entity,
            project=project,
            log_model=True,
            notes=config["description"],
            config=config,
        )

        self.run = self.wandb_logger.experiment
        self.wandb_run_id = self.wandb_logger.version

    def log_code(self, code_path: str, artifact_name: str):
        """Upload a source folder as the code artifact <artifact_name>-<run_id>."""
        artifact_name = f"{artifact_name}-{self.wandb_run_id}"
        self.run.log_code(code_path, artifact_name)

    def log_table_artifacts(self, data: Mapping[str, Mapping[str, pd.DataFrame]]):
        """Upload {artifact: {table_name: DataFrame}} as W&B table artifacts."""
        for artifact_name, artifact_data in data.items():
            artifact_name_ = f"{artifact_name}-{self.wandb_run_id}"
            artifact_type = artifact_name
            artifact = wandb.Artifact(artifact_name_, artifact_type)
            for data_name, df in artifact_data.items():
                table = dataframe_to_wandb_table(dataframe=df.reset_index())
                artifact.add(table, data_name)
            self.run.log_artifact(artifact)
