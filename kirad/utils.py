"""Shared helpers: per-recording averaging, W&B artifact I/O, dynamic imports, analysis.

Stages: launch (import_object, localize_wandb_artifact), loss (segment mean/variance),
outputs (cache_backbone, score dataframes, indeterminate analysis, report cards).

Various utility functions, such as label smoothing and computing thresholds.
"""
import configparser
import copy
import csv
import functools
import importlib
import json
import os
import tempfile
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from textwrap import dedent
from typing import Callable, Iterable, Literal, Optional, Sequence, Tuple

import git
import numpy as np
import pandas as pd
import torch
import torchaudio
import wandb
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kirad.constants import USER_CACHE_DIR_KINTSUGI, DatasetFields
from kirad.dataset_utils import load_relative_dataset
from kirad.metrics import IndetRocCurve, IndetSnSpArray


def load_audio(
    fp: str | Path, target_sample_rate: Optional[int] = None
) -> Optional[torch.Tensor]:
    """Load channel 0 of an audio file, optionally resampling; not used by the datasets.

    Load and resample an audio file.

    Returns None if the audio cannot be loaded or is empty.

    Parameters
    ----------
    fp : string or Path
        Filepath to the audio file to be loaded
    target_sample_rate : int or None
        Sample rate to resample the audio file to. If None, does not resample

    Returns
    -------
    torch.Tensor or None
        Returns the resampled audio file, or None if the audio cannot be loaded
        or is empty.
    """
    if isinstance(fp, str):
        fp = Path(fp)

    try:
        audio, sample_rate = torchaudio.load(fp)
    except RuntimeError as e:
        logger.warning(f"{fp.name} could not be loaded: {e}")
        return None

    if audio.dim() == 2:
        audio = audio[0].squeeze()

    if len(audio) == 0:
        logger.warning(f"{fp.name} is empty")
        return None

    if target_sample_rate is not None:
        if target_sample_rate > sample_rate:
            logger.warning(
                f"Upsampling {fp.name} from {sample_rate} to {target_sample_rate}"
            )
        audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)

    return audio


def average_tensor_in_segments(tensor: torch.Tensor, lengths: list[int] | torch.Tensor):
    """Mean over each recording's windows: [ΣW, ...] -> [B, ...] using `length`.

    Average segments of a tensor based on a list of lengths

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to average
    lengths : list of ints
        The lengths of each segment to average in the tensor, in order

    Returns
    -------
    torch.Tensor
        The tensor with relevant segments averaged
    """
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, device=tensor.device)
    index = torch.repeat_interleave(
        torch.arange(len(lengths), device=tensor.device), lengths
    )
    out = torch.zeros(
        lengths.shape + tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    out.index_add_(0, index, tensor)
    broadcastable_lengths = lengths.view((-1,) + (1,) * (len(out.shape) - 1))
    return out / broadcastable_lengths


def average_and_variance_tensor_in_segments(
    tensor: torch.Tensor, lengths: list[int] | torch.Tensor
):
    """Per-recording mean and (biased) variance of window scores: [ΣW, ...] -> two [B, ...].

    Average and variance of segments of a tensor based on a list of lengths

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to whose segments' averages and variances should be computed
    lengths : list of ints
        The lengths of each segment of in the tensor, in order

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        The tensors with relevant segments' averages and variances
    """
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, device=tensor.device)
    mean = average_tensor_in_segments(tensor, lengths)
    mean_broadcast = torch.repeat_interleave(mean, lengths, dim=0)
    return mean, average_tensor_in_segments(
        torch.square(tensor - mean_broadcast), lengths
    )


def localize_wandb_artifact(
    run: wandb.apis.public.Run,
    artifact_name: str,
    tag: str = "best",
    return_dir: bool = False,
):
    """Download a W&B artifact once to ~/.cache/kintsugi and return its local path.

    Download an artifact from wandb and return its local path.
    Note: returns the first file in the folder unless `return_dir`; cache is never refreshed.
    """
    artifact_dir = USER_CACHE_DIR_KINTSUGI / f"wandb_run_{run.id}" / artifact_name / tag

    # If the directory doesn't exist, the artifact was not cached before,
    # so we need to download it
    if not artifact_dir.exists():

        artifact_dir.mkdir(parents=True, exist_ok=True)

        wandb_name = f"{run.entity}/{run.project}/{artifact_name}:{tag}"

        artifact = wandb.Api().artifact(wandb_name)
        artifact.download(artifact_dir)

    return (
        artifact_dir if return_dir else artifact_dir / list(artifact_dir.iterdir())[0]
    )


def get_wandb_run_name_tag_from_artifact_url(
    url: str,
) -> Tuple[wandb.apis.public.Run, str, str]:
    """Parse a W&B artifact URL into (run that logged it, artifact name, tag).

    Resolves wandb run and name from artifact url

    Parameters
    ----------
    url : str
        Standard Wandb URL of the artifact, such as
        https://entity.wandb.io/entity/project/artifacts/model/model-run_id/best

    Returns
    -------
    run: `wandb.apis.public.Run` object corresponding to the given url
    name: Name of the artifact inside the run (without tag)
    tag: identifier for a specific model within this run; typically "best" or "v0", "v1", "v2", etc.

    """

    #
    ckpt_path = url.replace("https://", "").split("/")
    # HARDCODED: URL layout host/entity/project/artifacts/type/name[/tag]; tag defaults to "best"
    assert 6 <= len(ckpt_path) <= 7  # len is 6 when model tag is omitted

    #
    entity, project, name = ckpt_path[1], ckpt_path[2], ckpt_path[5]
    tag = "best" if len(ckpt_path) == 6 else ckpt_path[6]

    #
    api = wandb.Api()
    artifact_path = f"{entity}/{project}/{name}:{tag}"
    run = api.artifact(artifact_path).logged_by()

    #
    return run, name, tag


class TSVAccumulator:
    """Helper for formatting tab-separated-value data with section headers in the first row."""

    def __init__(self):
        """Start with an empty header and one empty row."""
        self.header = ""
        self.rows = [[]]

    def set_header(self, header: str):
        """Set the section prefix used by later `write` calls."""
        self.header = header

    def write(self, first: str, *rest: str) -> None:
        """Add one column: "<header> - <first>" in row 0, then `rest` down the rows."""
        entries = (f"{self.header} - {first}",) + rest
        for _ in range(len(entries) - len(self.rows)):
            self.rows.append([""] * len(self.rows[0]))
        padded_entries = entries + ("",) * (len(self.rows) - len(entries))
        for i, entry in enumerate(padded_entries):
            self.rows[i].append(entry)

    def __str__(self):
        """Render rows as TSV text."""
        buffer = StringIO()
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        writer.writerows(self.rows)
        return buffer.getvalue()


def get_wandb_report_card_tsvs(
    wandb_path: str,
    wandb_path_test: Optional[str] = None,
    threshold_objective: str = "macro_f1",
) -> dict[str, str]:
    """Build a TSV report card per task from a W&B run and log it back as an artifact.

    Create model report cards for each task, suitable for pasting into google sheets.
    Note: reads the "best" checkpoint's thresholds and test summary keys (sn_eq_sp@0%i, auroc@0%i).

    Parameters
    ---------
    wandb_path : output from the copy button on a wandb run overview page, e.g. "entity/project/run_id"
    threshold_objective : condition used for selecting model thresholds; allowable values depend on run

    Returns
    -------
    mapping from task name to tab-separated-value formatted report card for that task

    """
    wandb_path_test = wandb_path_test or wandb_path

    api = wandb.Api()
    run = api.run(wandb_path)
    run_test = api.run(wandb_path_test)

    artifact = wandb.Artifact(f"report-cards-{run_test.id}", "report-cards")

    data_params = run.config["data_params"]
    ckpt_path = localize_wandb_artifact(
        run,
        f"model-{run.id}",
        "best",  # Can't customize `tag` because the metrics are only computed for "best"
    )
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"]
    _, indet_oma = indet_analysis(wandb_path, wandb_path_test)

    tsvs = dict()
    for task in run.config["model_params"]["tasks"]:
        tsv = TSVAccumulator()
        tsv.set_header("Model info")
        tsv.write("name", "FILL IN")
        tsv.write("version", "FILL IN")
        tsv.write("deployed date", "FILL IN")
        tsv.write("wandb model registry url", "FILL IN")
        tsv.write("trained date", run.metadata["startedAt"].split("T")[0])
        tsv.write("wandb run url", run.url)
        tsv.set_header("Datasets")
        for split in ("train", "val", "test"):
            tsv.write(
                split,
                f"{data_params[split + '_split_or_dataset_version']} ({', '.join(data_params['data_sources'])})",
            )
        tsv.set_header("Binary OMA metrics")
        tsv.write(
            "sn_eq_sp_test",
            f"{run_test.summary[f'{task}/metrics/test/sn_eq_sp@0%i']:.1%}",
        )
        tsv.write(
            "auroc_test",
            f"{run_test.summary[f'{task}/metrics/test/auroc@0%i']:.1%}",
        )
        indet_task = indet_oma.loc[task].rename_axis("indet_budget_val").reset_index()
        for series_name, series in indet_task.items():
            tsv.write(series_name, *[f"{item:.1%}" for item in series])
        tsv.set_header("Multi-class metrics")
        for class_metric in (
            "absolute_error",
            "macro_f1",
            "macro_recall",
            "macro_precision",
            "accuracy",
        ):
            val = run_test.summary[
                f"{task}/thresh_metrics/test/{class_metric}/tuned_for_{threshold_objective}"
            ]
            baseline = float("nan")
            min_max = "⬇️" if class_metric == "absolute_error" else "⬆️"
            # HARDCODED: looks for baseline_const_0..3 only (tasks with up to 4 classes)
            for const in range(4):
                try:
                    baseline = run_test.summary[
                        f"{task}/thresh_metrics/test/{class_metric}/baseline_const_{const}"
                    ]
                except KeyError:
                    pass
            tsv.write(f"{class_metric} {min_max}", f"{val:.3f}")
            tsv.write("baseline for ⬅️", f"{baseline:.3f}")
        tsv.set_header("Multi-class thresholding")
        tsv.write("objective", threshold_objective)
        tsv.write(
            "thresholds",
            str(
                state_dict[
                    f"tasks.{task}.thresholding.{threshold_objective}.thresholds"
                ].tolist()
            ),
        )
        tsvs[task] = str(tsv)
        # Need .txt extension for wandb to display contents inline
        with artifact.new_file(f"{task}.tsv.txt", "w", "utf-8") as f:
            f.write(tsvs[task])

    entity, project, id_ = wandb_path_test.split("/")
    with wandb.init(
        project=project, entity=entity, id=id_, resume="allow"
    ) as resumed_run:
        resumed_run.log_artifact(artifact)
    return tsvs


def get_wandb_report_card_tsvs_with_instructions(
    wandb_path: str,
    wandb_path_test: Optional[str] = None,
    threshold_objective="macro_f1",
) -> str:
    """Report cards from get_wandb_report_card_tsvs, prefixed with paste instructions.

    Create model report cards for each task, and instructions for pasting into google sheets.

    Parameters
    ---------
    wandb_path : output from the copy button on a wandb run overview page, e.g. "entity/project/run_id"
    threshold_objective : condition used for selecting model thresholds; allowable values depend on run

    Returns
    -------
    String with report cards and instructions for pasting these into google sheets.

    """
    tsvs = get_wandb_report_card_tsvs(
        wandb_path, wandb_path_test, threshold_objective=threshold_objective
    )
    header = dedent(
        """\
        Paste each of these model report card TSVs into the tab of the google sheet

        Process:
          (1) Select TSV for a task and copy.
          (2) Select the first empty cell in the first column of the corresponding tab of the google sheet and paste.
          (3) Verify the new row of column headers lines up with the one in the first row.
          (4) Delete the new row of column headers so the new model's stats are below the previous model's.
          (5) Fill in the "FILL IN" values manually.

    """
    )
    task_text = "Task: {task} (TSV for pasting into spreadsheet)\n\n{tsv}\n\n"
    return header + "".join(
        task_text.format(task=task, tsv=tsv) for task, tsv in tsvs.items()
    )


def str_to_tensor(s: str):
    """Encode a python string as 1-d `torch.Tensor` of `uint8`s."""
    return torch.frombuffer(bytearray(s, encoding="utf-8"), dtype=torch.uint8)


def strs_to_tensor(strs: Sequence[str]):
    """Encode strings as a zero-padded 2-d uint8 tensor (uuids for MultiCatMetric).

    Encode a sequence of python strings of equal length as a 2-d `torch.Tensor` of `uint8`s."""
    return torch.nn.utils.rnn.pad_sequence(
        tuple(map(str_to_tensor, strs)), batch_first=True
    )


def tensor_to_str(t: torch.Tensor) -> str:
    """Decode a 1-d `torch.Tensor` of `uint8`s into a python string (strips zero padding)."""
    return bytes(t.cpu().numpy().data).decode("utf-8").rstrip("\x00")


def tensor_to_strs(t: torch.Tensor) -> list[str]:
    """Decode a 2-d `torch.Tensor` of `uint8`s into a list of python strings."""
    return list(map(tensor_to_str, t))


def get_dataframe_with_scores(
    wandb_path: str,
    split: Literal["val", "test"],
    scores_tag: Optional[str] = None,
    model_tag: Optional[str] = None,
) -> pd.DataFrame:
    """Join a run's metadata CSVs with its W&B scores-<task>-<split> tables on uuid.

    Fetches val or test metadata used for a training run and joins with the scores computed during training.

    Arguments
    ---------
    wandb_path : run path copied from wandb run overview page, e.g. "entity/project/run_id"
    split : which of "val" or "test" to fetch data for
    scores_tag : wandb tag specifying which epoch's scores to use. Can only specify one of `scores_tag` or `model_tag`.
        By default, gets scores from `best`-tagged model. Ignored if split == "test" since test scores are only computed
        once.
    model_tag : wandb tag specifying which model to get scores of. Can only specify one of `scores_tag` or `model_tag`.
        By default, gets scores from `best`-tagged model. Ignored if split == "test" since test scores are only computed
        once.

    Returns
    -------
    Dataframe of metadata with two additional columns for each task model was trained on:
        scores_{task} : model scores on that task
        quantified_labels_{task} : quantfied labels used during eval, e.g. takes values 0, 1, 2 for a 3-class model

    """
    api = wandb.Api()
    run = api.run(wandb_path)
    if split == "test":
        if scores_tag not in (None, "v0") or model_tag not in (None, "best"):
            raise ValueError(
                f"Test split is only evaluated once; invalid combination of {scores_tag=} and {model_tag=}"
            )
        scores_tag = "v0"  # HARDCODED: test scores are logged once, as version v0
    else:
        if scores_tag is not None and model_tag is not None:
            raise ValueError(
                f"Got {scores_tag=} and {model_tag=}; at most one should be specified."
            )
        if scores_tag is None:
            model_tag = model_tag or "best"
            ckpt_path = localize_wandb_artifact(run, f"model-{run.id}", model_tag)
            model_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
            scores_tag = f"v{model_dict['epoch']}"

    # While it appears load_dataset should only need to be run for `split` and
    # not for all splits, currently
    # OrdinalRegressionPLModule.on_validation_test_epoch_end saves scores under
    # a split name based on whether the PL Trainer is running in `validate` or
    # `test` mode, not based on the actual dataset split. So even if the
    # data_params.eval_set config argument is set to something besides "test"
    # (e.g. to "train" for knowledge distillation), the scores will still be
    # saved under the name "test". Concatenating the base dataset over all
    # splits works around this issue, so the columns coming from the base
    # dataset will be present for the rows for which scores are present,
    # regardless of which split these actually came from.
    # TODO Clean this up somehow so no hack is needed.
    data_params = run.config["data_params"]
    new_format = (
        "metadata" in data_params
        or "metadata=kirad.datasets_and_dataloaders.KintsugiMetadata" in data_params
    )
    if new_format:
        data_params = data_params.get("metadata") or data_params.get(
            "metadata=kirad.datasets_and_dataloaders.KintsugiMetadata"
        )
        # TODO Refactor LauncherConfig to pull out get_params_by_split and use it here
        for s in ["train", "val", "test"]:
            split_params = data_params.get(s, {})
            data_params[s] = copy.deepcopy(data_params.get("default", {}))
            if s in {"val", "test"}:
                data_params[s] |= data_params.get("val_test", {})
            data_params[s] |= split_params

    #
    dataset_version, data_sources, dataset_folds = {}, {}, {}
    for s in ["train", "val", "test"]:
        if new_format:
            dataset_version[s] = data_params[s]["df_or_dataset_version"]
            dataset_folds[s] = data_params[s].get("dataset_fold")
            data_sources[s] = data_params[s].get("data_sources")
        else:
            dataset_version[s] = data_params[f"{s}_split_or_dataset_version"]
            dataset_folds[s] = data_params["dataset_fold"]
            data_sources[s] = data_params["data_sources"]
        if not isinstance(dataset_version[s], list):
            dataset_version[s] = [dataset_version[s]]
    #
    df = pd.concat(
        [
            load_relative_dataset(
                d,
                data_sources[s],
                s,
                dataset_folds[s],
                audio_root=data_params.get("audio_root"),
                text_root=data_params.get("text_root"),
            )
            for s in ["train", "val", "test"]
            for d in dataset_version[s]
        ]
    )
    # This is needed if you specify your datasets as csv files.
    # In such case you can provide same .csv file for, let's say, val and test splits.
    # This may lead to duplicate entries/rows (or even three-plicates).
    df.drop_duplicates(inplace=True)

    if new_format:
        tasks = run.config["model_params"]["config"]["classifier_config"][
            "tasks"
        ].keys()
    else:
        tasks = run.config["model_params"]["tasks"].keys()
    for task in tasks:
        task_df = (
            api.artifact(
                f"{run.entity}/{run.project}/scores-{task}-{split}-{run.id}:{scores_tag}",
                type="scores",
            )
            .get("scores")
            .get_dataframe()
        )
        task_df.rename(
            columns={
                "scores": f"scores_{task}",
                "score_variances": f"score_variances_{task}",
                "quantized_labels": f"quantized_labels_{task}",
            },
            inplace=True,
        )
        task_df.drop(columns="labels", inplace=True)
        df = df.merge(task_df, on="uuid", validate="1:1")
    return df


def indet_analysis(
    wandb_path: str,
    wandb_path_test: Optional[str] = None,
    budgets: Iterable[float] = tuple(np.linspace(0.0, 0.9, 10)),  # HARDCODED: budgets 0%..90%
):
    """Tune sn_eq_sp thresholds on val per budget and binary cut, then measure them on test.

    Returns (per-cut records, mean over cuts per task and budget) as DataFrames.
    Hardcoded: tasks ("anxiety", "depression"); budgets 0.0..0.9 in steps of 0.1.
    """
    df_val = get_dataframe_with_scores(wandb_path, "val")
    df_test = get_dataframe_with_scores(wandb_path_test or wandb_path, "test")
    records = []
    for task in ("anxiety", "depression"):  # HARDCODED: task names; edit for other outcomes
        labels_val = df_val[f"quantized_labels_{task}"]
        scores_val = df_val[f"scores_{task}"]
        labels_test = df_test[f"quantized_labels_{task}"]
        scores_test = df_test[f"scores_{task}"]
        for quantized_label_thresh in range(labels_val.max()):
            issa = IndetSnSpArray.build(
                y_true=(labels_val > quantized_label_thresh).astype(int),
                y_score=scores_val,
            )
            for budget in budgets:
                roc = issa.roc_curve(budget)
                op = roc.sn_eq_sp()
                op_on_test = op.eval(
                    y_true=(labels_test > quantized_label_thresh).astype(int),
                    y_score=scores_test,
                )
                records.append(
                    dict(
                        task=task,
                        quantized_label_thresh=quantized_label_thresh,
                        indet_budget_val=budget,
                        sn_eq_sp_val=op.min_sn_sp,
                        auroc_val=roc.auc(),
                        lower_thresh=op.lower_thresh,
                        upper_thresh=op.upper_thresh,
                        sn_test=op_on_test.sn,
                        sp_test=op_on_test.sp,
                        indet_frac_diff=op_on_test.indet_frac - budget,
                    )
                )
    df_indet = pd.DataFrame(records)
    gb = df_indet.groupby(["task", "indet_budget_val"])
    gb_oma_cols = gb[
        ["sn_eq_sp_val", "auroc_val", "sn_test", "sp_test", "indet_frac_diff"]
    ]
    df_oma_indet = gb_oma_cols.mean()
    return df_indet, df_oma_indet


def analyze_dataframe_with_scores(
    df: pd.DataFrame,
    quantize_and_groupby: dict[str, Optional[Sequence]],
    tasks: Iterable[str] = ("anxiety", "depression"),
):
    """Mean AUROC and sn_eq_sp (0% indet) over binary cuts, per subgroup of the dataframe.

    Group dataframe by specified columns (possibly quantizing), computing metrics for each group.
    Note: adds `quantized_<key>` columns to `df` in place.

    Arguments
    ---------
    df : DataFrame containing columns for each key of `quantize_and_groupby` as well as `f"quantized_labels_{task}"`
        and `f"scores_{task}"` for each `task` in `tasks` and a row for each scored utterance in the dataset
    quantize_and_groupby : dictionary whose keys are the column names by which rows of `df` should be grouped and whose
        values are either `None` if the column values should be used as is or a sequence of bin endpoints if the column
        values should be binned before grouping
    tasks : names of tasks to analyze scores and labels for, e.g. `("anxiety", "depression")`

    """
    for key, bins in quantize_and_groupby.items():
        if bins is None:
            df[f"quantized_{key}"] = df[key]
        else:
            df[f"quantized_{key}"] = pd.cut(df[key], bins, right=False)
    num_splits = {task: df[f"quantized_labels_{task}"].max() for task in tasks}
    rows = []
    for vals, sub_df in df.groupby(
        [f"quantized_{key}" for key in quantize_and_groupby], observed=True
    ):
        row = dict(zip(quantize_and_groupby, vals))
        for task in tasks:
            auroc_sum = 0
            sn_eq_sp_sum = 0
            for c_idx in range(num_splits[task]):
                roc_curve = IndetRocCurve.build(
                    y_true=(sub_df[f"quantized_labels_{task}"] > c_idx).astype(int),
                    y_score=sub_df[f"scores_{task}"],
                )
                auroc_sum += roc_curve.auc()
                sn_eq_sp_sum += roc_curve.sn_eq_sp().min_sn_sp
            row["n"] = len(sub_df)
            row[f"auroc_{task}"] = auroc_sum / (num_splits[task])
            row[f"sn_eq_sp_{task}"] = sn_eq_sp_sum / (num_splits[task])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def get_package_path(package: Literal["kipy", "kirad"]) -> Path:
    """Repo root of kipy or kirad (env KIPY_PATH / KINTSUGI_RAD_PATH, else from import).

    Get local paths to `kintsugi-rad` and `kipy`.

    Use the paths in the environment variables KIPY_PATH and KINTSUGI_RAD_PATH if
    specified. Otherwise, first find the paths to the `kipy` and `kirad` packages. The
    packages `kipy` and `kirad` are installed one level under `kipy` and `kintsugi-rad`
    respectively, so set the paths to `kipy` and `kintsugi-rad` as the parent
    directories.

    Parameters
    ----------
    package : str
        Name of the package to get the local path.

    Raises
    ------
    ValueError
        Raised if `package` isn't "kipy" or "kirad".
    """
    if package == "kipy":
        os_env_var = "KIPY_PATH"
    elif package == "kirad":
        os_env_var = "KINTSUGI_RAD_PATH"
    else:
        raise ValueError(f'Package must be "kipy" or "kirad". Received "{package}".')

    package_path = os.getenv(os_env_var)
    if package_path is None:
        # Not set in the environment
        # importlib.util.find_spec(package).origin returns /path/to/package/__init__.py,
        # so go 2 levels up to get the paths to `kintsugi-rad` and `kipy`.
        package_path = (
            Path(importlib.util.find_spec(package).origin).resolve().parent.parent
        )

    return Path(package_path)


def get_package_info(package: Literal["kipy", "kirad"]) -> Mapping[str, str]:
    """Version (from setup.cfg) and git commit/dirty state of kipy or kirad.

    Get package version and git commit hash for kipy and kirad.
    Note: raises if the package is not importable, e.g. the private kipy.

    Parameters
    ----------
    package : str
        Name of the package to get the version and commit hash.

    Returns
    -------
    dict : contains the package name, package version, git commit hash, and whether the
        repo is in a dirty state.
    """
    package_path = get_package_path(package)

    # Get the package version. Querying the version using `importlib.metadata.version`
    # gives the version of the package when it was installed (using `pip install -e .`),
    # so it might not match the version in the current working branch. So instead, get
    # the version from `setup.cfg` directly.
    package_config = configparser.ConfigParser()
    package_config.read(package_path / "setup.cfg")
    package_version = package_config["metadata"]["version"]

    # Get git info: commit hash and repo dirty state.
    try:
        repo = git.Repo(package_path)
        if repo.bare:
            # Empty git repo
            commit_hash = "Empty git repository"
            is_dirty = None
        else:
            commit_hash = repo.head.object.hexsha
            is_dirty = repo.is_dirty()
    except git.exc.InvalidGitRepositoryError:
        commit_hash = "Not a valid git repository"
        is_dirty = None
        logger.warning(
            f'Package path "{package_path}" for package "{package}" is not a valid '
            "git repo. Will not log git info for this package."
        )

    return {
        "package": package,
        "version": package_version,
        "commit_hash": commit_hash,
        "is_dirty": is_dirty,
    }


def import_object(object_path: str) -> type:
    """Import a class/function from a dotted path, e.g. "model.DepAnxClassifier".

    Return a reference to the object specified by the object path.

    Parameters
    ----------
    object_path : str
        Path to an object to load. Can be a path inside a package or a python file
        that is in the python system path.
        Examples of paths inside a package:
            "kipy.models.e2e.whisper_ft_model.WhisperFTModel"
            "kirad.base_models.WhisperBackbone"
        Example of loading "DepAnxClassifer" inside model.py that is in the system path:
            "model.DepAnxClassifier"

    Returns
    -------
    type[Any]
        The object. Note that this is not an instantiation of the object; is a reference
        to the object itself.
    """
    pkg, obj = object_path.rsplit(".", maxsplit=1)
    module = importlib.import_module(pkg)

    return getattr(module, obj)


def transfer_to_cuda(x, device=0):
    """Move a tensor or a (nested) dict of tensors to a CUDA device."""
    if isinstance(x, torch.Tensor):
        return x.cuda(device)

    #
    if isinstance(x, Mapping):
        return {k: transfer_to_cuda(v, device) for k, v in x.items()}
    else:
        raise TypeError("Unsupported nested type")


def cache_backbone(
    backbone: Callable,
    wandb_logger: Optional[WandbLogger],
    datasets: list[Dataset],
    data_loader_args: list[dict],
    state_dict: dict,
) -> dict:

    """Run the backbone on all datasets, gather outputs by uuid across ranks, log to W&B.

    Runs inference on a backbone and stores it, along with inference results,
    into wandb by using current wandb_logger session.

    Parameters
    ----------
    backbone: Callable
        Callable object, which receives 2 inputs -- (x, lengths) and returns backbone output.
        Underlying model should be set to a proper device before calling cache_backbone()
        Outputs will be on cpu.

    wandb_logger: Optional[WandbLogger]
        Wandb logger is supposed to be valid for global_rank == 0 node.
        See _setup_launcher function for details.

    datasets: list[Dataset]
        List of Dataset objects to iterate through and cache

    data_loader_args: list[dict]
        Arguments of the dataloaders

    state_dict: dict
        Backbone's state_dict with the correct param names with respect
        to the root module in the object hierarchy.
        For example: if the original backbone module has the object hierarchy
        model.my_submodule.my_backbone, then the keys in the state dict should start with
        the prefix "my_submodule.my_backbone."

    Returns
    -------
    dict : State dict which contains backbone porams and inference cache written under
    'backbone_cache' key
    """

    # sanity check
    assert len(datasets) == len(data_loader_args)
    data_loader_args = map(copy.deepcopy, data_loader_args)

    #
    local_cache = dict()
    for dataset, args in zip(datasets, data_loader_args):
        args["shuffle"], args["drop_last"] = False, False

        dataloader = DataLoader(dataset, **args)

        with (torch.inference_mode()):
            for batch in tqdm(dataloader):
                x = transfer_to_cuda(
                    batch[DatasetFields.FEATURES], torch.distributed.get_rank()
                )
                out = backbone(x, batch[DatasetFields.LENGTH])

                for uuid, x in zip(batch[DatasetFields.UUID], out, strict=True):
                    local_cache[uuid] = x.cpu()

        # make sure that all ranks finished their processing
        torch.distributed.barrier()

    logger.info(
        f"Rank: {torch.distributed.get_rank()}, Local cache size: {len(local_cache)}"
    )

    # sync and merge local caches
    global_cache = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(global_cache, local_cache)

    global_cache = functools.reduce(lambda a, b: a | b, global_cache)
    logger.info(f"Global cache size: {len(global_cache)}")

    backbone_dict = dict(backbone_cache=global_cache, state_dict=state_dict)

    # wandb_logger exists only for global_rank == 0 worker
    # see _setup_launcher() for details
    if wandb_logger:
        wandb_run_id = wandb_logger.version
        path = Path(tempfile.mkdtemp()) / "cache.pth"
        torch.save(backbone_dict, path)
        artifact = wandb.Artifact("backbone_cache-" + wandb_run_id, "backbone_cache")
        artifact.add_file(str(path))
        wandb_logger.experiment.log_artifact(artifact)

    return backbone_dict


def read_whisper_timestamped_json(json_file: str | Path) -> str:
    """Transcript text ("text" field) from a whisper-timestamped JSON.

    Get the text transcription from the JSON file output by whisper-timestamped.

    Parameters
    ----------
    json_file : str or Path
        Path to a JSON file output by whisper-timestamped

    Returns
    -------
    str
        The transcription.
    """
    with open(json_file) as jsonfile:
        data = json.load(jsonfile)

    return data["text"]


def read_google_asr_json(json_file: str | Path) -> str:
    """Transcript from a Google Speech API JSON: top alternative of each result, joined.

    Get the text transcription from the JSON file output by the Google Speech API.

    Parameters
    ----------
    json_file : str or Path
        Path to a JSON file output by whisper-timestamped

    Returns
    -------
    str
        The transcription.
    """
    with open(json_file) as jsonfile:
        data = json.load(jsonfile)

    return "".join(
        [
            result["alternatives"][0]["transcript"]
            for result in data["results"]
            if "alternatives" in result
        ]
    )


def flat_to_nested_dict(d_in: Mapping, sep: str = "."):
    """Split dot-separated keys into nested dicts.

    Convert a flat dict with dot-separated keys to a nested dict so e.g. `d_out['a']['b']['c'] = d_in['a.b.c']`.

    >>> d_in = {"first.second.third" : 3, "first.other": 4}
    >>> d_out = flat_to_nested_dict(d_in)
    >>> assert d_out == {"first": {"second": {"third": 3}, "other": 4}}

    """
    d_out = dict()
    for k, v in d_in.items():
        levels = k.split(sep)
        cur_level = d_out
        for i in range(len(levels) - 1):
            cur_level = cur_level.setdefault(levels[i], dict())
            if not isinstance(cur_level, dict):
                k_short = sep.join(levels[: i + 1])
                raise ValueError(f"key {k_short!r} reused as prefix of key {k!r}.")
        if levels[-1] in cur_level:
            k_long = next(key for key in d_in if key.startswith(k + sep))
            raise ValueError(f"key {k!r} reused as prefix of key {k_long!r}.")
        cur_level[levels[-1]] = v
    return d_out


def nested_to_flat_dict(
    d_in: Mapping, d_out: Optional[Mapping] = None, prefix: Optional[str] = None
):
    """Flatten nested dicts into dot-separated keys.

    Convert a nested dict to a flat one with dot-separated keys so e.g. `d_out['a.b.c'] = d_in['a']['b']['c']`.

    >>> d_in = {"first": {"second": {"third": 3}, "other": 4}}
    >>> d_out = nested_to_flat_dict(d_in)
    >>> assert d_out == {"first.second.third" : 3, "first.other": 4}
    True

    """
    if d_out is None:
        d_out = dict()
    if isinstance(d_in, dict):
        for key, value in d_in.items():
            if prefix is not None:
                key = f"{prefix}.{key}"
            nested_to_flat_dict(value, d_out, key)
    else:
        d_out[prefix] = d_in
    return d_out


def combine_backbone_modality_state_dicts(**backbone_dicts: Mapping):
    """Merge backbones of several checkpoints into one DAM 2 state_dict (backbone.<key>.*).

    Create a DAM 2 style backbone-only state_dict from the backbones of the given DAM 1 or 2 style state_dicts.

    e.g. `combine_backbone_state_dicts(audio=dam_1_state_dict, text=dam_2_state_dict)`.

    """
    merged = dict()
    for key, backbone_dict in backbone_dicts.items():
        # The weights may or may not be nested under `"state_dict"`
        backbone_dict = backbone_dict.get("state_dict", backbone_dict)
        backbone = flat_to_nested_dict(backbone_dict)["backbone"]
        # The weights may (DAM 2) or may not (DAM 1) be nested under `key`
        merged[key] = backbone.get(key, backbone)
    return dict(state_dict=nested_to_flat_dict(dict(backbone=merged)))


def combine_backbone_modality_state_dicts_from_wandb(
    output_filename: str, **backbone_run_ids: str
):
    """Save a merged DAM 2 backbone checkpoint built from the "best" models of W&B runs.

    Create a checkpoint for a DAM 2 style backbone-only state_dict from wandb run ids of individual backbones.
    """
    api = wandb.Api()
    torch.save(
        combine_backbone_modality_state_dicts(
            **{
                key: torch.load(
                    localize_wandb_artifact(api.run(run_id), f"model-{run_id}")
                )
                for key, run_id in backbone_run_ids.items()
            }
        ),
        output_filename,
    )


def dataframe_to_wandb_table(*, dataframe, **kwargs):
    """wandb.Table from a DataFrame, raising W&B's row limit so large tables are not cut."""
    wandb.Table.MAX_ARTIFACT_ROWS = max(wandb.Table.MAX_ARTIFACT_ROWS, len(dataframe))
    return wandb.Table(dataframe=dataframe, **kwargs)


def open_single_file_wandb_artifact(artifact: wandb.Artifact):
    """Download the only file of a W&B artifact and return an open handle."""
    (file,) = iter(artifact.files())
    # Ideally we'd use exist_ok=True instead of replace=True, but wandb has a bug wherein
    # exist_ok uses files from other runs instead of downoading the file for the correct run
    return file.download(replace=True)


def wandb_table_artifact_to_dataframe(artifact: wandb.Artifact):
    """DataFrame from a single-table W&B artifact, read directly from its JSON.

    Get dataframe from a wandb artifact containing a single table much faster than using `artifact.get`.
    """
    with open_single_file_wandb_artifact(artifact) as f:
        table_dict = json.load(f)
    return pd.DataFrame.from_records(table_dict["data"], columns=table_dict["columns"])
