"""Contains utility functions for processing and manipulating datasets, primarily in
pandas DataFrame format.
"""

import copy
import os
import subprocess
import typing
from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import entropy
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# torch does not seem to expose this module in a user-facing interface -- if it moves we will have to change this
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from .constants import (
    AUDIO_DIR,
    DATASET_VERSION_METADATA,
    FEATURE_STORE_URI,
    METADATA_DIR,
    RAW_AUDIO_URI,
    DatasetFields,
    DatasetSplit,
    DatasetVersion,
    DataSource,
    FeatureStoreBucket,
)

DistributionData = namedtuple(
    "DistributionData", ["distribution", "bin_edges", "categories"]
)


def load_dataset(
    dataset: pd.DataFrame | DatasetVersion | str | Path | Mapping,
    data_sources: Optional[DataSource | list[DataSource]] = None,
    dataset_splits: Optional[DatasetSplit | list[DatasetSplit]] = None,
    dataset_fold: Optional[int] = None,
) -> pd.DataFrame:
    if isinstance(dataset, pd.DataFrame):
        return dataset
    elif dataset in typing.get_args(DatasetVersion):
        # User provided a dataset version
        if (
            data_sources is None
            or dataset_splits is None
            or (dataset_splits != "test" and dataset_fold is None)
        ):
            raise ValueError(
                "To use a dataset from `kintsugi-ml-datasets`, pass in "
                "`data_sources`, `dataset_splits`, and `dataset_fold`"
            )
        if isinstance(data_sources, str) or not isinstance(data_sources, Iterable):
            data_sources = [data_sources]
        if isinstance(dataset_splits, str) or not isinstance(dataset_splits, Iterable):
            dataset_splits = [dataset_splits]
        full_dataset = load_dataset_version(
            dataset, data_sources, dataset_splits, dataset_fold
        )
        return pd.concat(
            [
                full_dataset[ds][split][0]
                for split in dataset_splits
                for ds in data_sources
            ],
            ignore_index=True,
        )
    elif isinstance(dataset, str) or isinstance(dataset, Path):
        # User provided a string. Assume this is a path to CSV file.
        return pd.read_csv(dataset)
    else:
        raise ValueError(f"Expected a DataFrame or a string; received {type(dataset)}")


def load_relative_dataset(
    dataset: pd.DataFrame | DatasetVersion | str | Path | Mapping,
    data_sources: Optional[DataSource | list[DataSource]] = None,
    dataset_splits: Optional[DatasetSplit | list[DatasetSplit]] = None,
    dataset_fold: Optional[int] = None,
    audio_root: Optional[str | Path] = None,
    text_root: Optional[str | Path] = None,
) -> pd.DataFrame:
    if isinstance(dataset, Mapping):
        audio_root = dataset.get("audio_root", audio_root)
        text_root = dataset.get("text_root", text_root)
        dataset = dataset["metadata"]
    df = load_dataset(dataset, data_sources, dataset_splits, dataset_fold)
    audio_root = audio_root or os.getcwd()
    text_root = text_root or os.getcwd()
    df["audio_filename"] = df["filename"].map(
        lambda s: os.path.expanduser(os.path.join(audio_root, str(s) + ".wav"))
    )
    df["text_filename"] = df["filename"].map(
        lambda s: os.path.expanduser(os.path.join(text_root, str(s) + ".json"))
    )
    return df


def load_dataset_version(
    dataset_version: DatasetVersion,
    data_sources: Optional[DataSource | list[DataSource]] = None,
    dataset_splits: Optional[DatasetSplit | list[DatasetSplit]] = None,
    dataset_folds: Optional[int | list[int]] = None,
) -> dict[DataSource, dict[DatasetSplit, list[pd.DataFrame]]]:
    if dataset_version not in typing.get_args(DatasetVersion):
        raise ValueError(
            f"Dataset version {dataset_version} is not registered kirad/constants.py."
        )
    if dataset_version not in DATASET_VERSION_METADATA:
        raise ValueError(
            f"Metadata for dataset version {dataset_version} is not defined in"
            "kirad/constants.py."
        )

    if data_sources is None or data_sources == "all" or "all" in data_sources:
        data_sources = [i for i in typing.get_args(DataSource) if i != "all"]
    elif isinstance(data_sources, str) or not isinstance(data_sources, Iterable):
        data_sources = [data_sources]
    if dataset_splits is None or dataset_splits == "all" or "all" in dataset_splits:
        dataset_splits = [i for i in typing.get_args(DatasetSplit) if i != "all"]
    elif isinstance(dataset_splits, str) or not isinstance(dataset_splits, Iterable):
        dataset_splits = [dataset_splits]
    if dataset_folds is None:
        dataset_folds = list(
            range(0, DATASET_VERSION_METADATA[dataset_version]["n_folds"])
        )
    elif not isinstance(dataset_folds, Iterable):
        dataset_folds = [dataset_folds]

    # Download metadata
    # When the destination directory already exists or when a disk is mounted in a
    # read-only mode don't try to copy the data. It is assumed the data is already
    # there and have the correct directory structure.
    out_dir = METADATA_DIR / dataset_version
    dir_exists = out_dir.exists()
    if not dir_exists:
        out_dir.mkdir(parents=True)
    stat = os.statvfs(out_dir)
    read_only = bool(stat.f_flag & os.ST_RDONLY)
    if not dir_exists and not read_only:
        if DATASET_VERSION_METADATA[dataset_version]["uri"] is None:
            raise ValueError(
                f"Need to define URI for dataset version {dataset_version}."
            )
        cmd = [
            "gsutil",
            "-m",
            "rsync",
            "-r",
            DATASET_VERSION_METADATA[dataset_version]["uri"],
            out_dir,
        ]
        try:
            subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, universal_newlines=True
            )
        except subprocess.CalledProcessError as e:
            logger.info(e.output)
            raise e
    else:
        if dir_exists:
            logger.info("Dataset directory already exists. Skipping metadata download.")
        else:
            logger.info("Disk is in read-only mode. Skipping metadata download.")

    # Gather the data across the data source and splits
    full_dataset = dict()
    for data_source in data_sources:
        full_dataset[data_source] = {"train": [], "val": [], "test": []}
        for dataset_split in dataset_splits:
            if dataset_split == "test":
                dataset_fold_strs = [""]
            else:
                # Train and val splits
                dataset_fold_strs = [f".fold_{fold}" for fold in dataset_folds]
            for fold_str in dataset_fold_strs:
                data_csv = (
                    out_dir / "splits" / f"{data_source}.{dataset_split}{fold_str}.csv"
                )
                try:
                    tmp_df = pd.read_csv(data_csv).drop_duplicates(
                        subset="filename", ignore_index=True
                    )
                    tmp_df["split"] = dataset_split
                    full_dataset[data_source][dataset_split].append(tmp_df)
                except FileNotFoundError as e:
                    logger.warning(e)

    return full_dataset


def resolve_audio_path(
    audio_dir_or_feature_store_bucket: FeatureStoreBucket | str | Path,
    is_raw_audio: bool = False,
    force_download: bool = False,
) -> Path:
    if audio_dir_or_feature_store_bucket in typing.get_args(FeatureStoreBucket):
        return resolve_audio_path_from_bucket(
            audio_dir_or_feature_store_bucket,
            is_raw_audio=is_raw_audio,
            force_download=force_download,
        )
    elif isinstance(audio_dir_or_feature_store_bucket, str | Path):
        return Path(audio_dir_or_feature_store_bucket)
    else:
        raise ValueError(
            f"Expected Path or a string; received {type(audio_dir_or_feature_store_bucket)}"
        )


def resolve_audio_path_from_bucket(
    feature_store_bucket: FeatureStoreBucket,
    is_raw_audio: bool = False,
    force_download: bool = False,
) -> Path:
    if feature_store_bucket not in typing.get_args(FeatureStoreBucket):
        raise ValueError(
            f"Received feature store bucket {feature_store_bucket}; "
            f"available dataset versions are {typing.get_args(FeatureStoreBucket)}"
        )

    audio_dir = get_cache_dir(feature_store_bucket)
    if not audio_dir.exists() or force_download:
        if FEATURE_STORE_URI is None:
            raise ValueError("Need to define URI for `FEATURE_STORE_URI`.")
        if RAW_AUDIO_URI is None:
            raise ValueError("Need to define URI for `RAW_AUDIO_URI`.")
        download_uri = (
            RAW_AUDIO_URI
            if is_raw_audio
            else f"{FEATURE_STORE_URI}/{feature_store_bucket}"
        )
        audio_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["gsutil", "-m", "rsync", "-r", download_uri, audio_dir]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)

    return audio_dir


def get_cache_dir(location: FeatureStoreBucket | str | Path) -> Path:
    if location in typing.get_args(FeatureStoreBucket):
        return AUDIO_DIR / location
    elif isinstance(location, str | Path):
        return Path(location)
    else:
        raise ValueError(
            f"Received feature store bucket {location}; "
            f"available dataset versions are {typing.get_args(FeatureStoreBucket)}"
        )


def calculate_distributions(
    df: pd.DataFrame,
    label_column: str,
    other_columns: Optional[list[str]] = None,
    existing_label_distribution: Optional[DistributionData] = None,
    existing_other_distributions: Optional[dict[str, DistributionData]] = None,
) -> tuple[DistributionData, Optional[dict[str, DistributionData]]]:
    """Calculate distributions for the labels and specified columns in the data

    The distributions will be density-normalized histograms for real-valued data, and
    probability masses for categorical data. If existing distributions are specified,
    then the bin edges/categories from those distributions will be used for calculating
    the distributions.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame containing the dataset.
    label_column : str
        Column in the DataFrame corresponding to the target labels (eg. "bin_phq"). For
        now, assumes that the labels are categorical and encoded as non-negative
        integers.
    other_columns : list[str], optional
        List of column names in the DataFrame for which to calculate distributions.
    existing_label_distribution : DistributionData, optional
        Contains distribution information about the labels. If provided, the calculated
        distributions will use the same bin edges/categories.
    existing_other_distributions : dict[str, DistributionData], optional
        Contains distribution information about various columns in the DataFrame, where
        the dictionary key specifies the column name. If provided, the calculated
        distributions will use the same bin edges/categories.

    Returns
    -------
    label_distribution : DistributionData
        Contains distribution information about the labels
    other_distributions : dict[str, DistributionData]
        Contains distribution information about various columns in the DataFrame, where
        the dictionary key specifies the column name. Will be None if other_columns is
        None.

    Raises
    ------
    ValueError
        Raised if one of the specified other columns does not exist in the dataset.
    """
    # Calculate label distribution
    # NOTE: this assumes categorical labels encoded as integers (0, 1, ...)
    # TODO: make this generalize to arbitrary categorical labels, and continuous labels
    if existing_label_distribution is None:
        min_length = 0
    else:
        min_length = len(existing_label_distribution.categories)
    count = np.bincount(df[label_column], minlength=min_length)
    distribution = count / np.sum(count)
    categories = list(range(len(count)))
    label_distribution = DistributionData(distribution, None, categories)

    # Calculate other distributions
    other_distributions = None
    if other_columns is not None:
        other_distributions = dict()
        for column in other_columns:
            data = df[column].dropna()
            data_type = get_data_type(data)
            if data_type == "numerical":
                if (
                    existing_other_distributions is not None
                    and column in existing_other_distributions
                ):
                    bins = existing_other_distributions[column].bin_edges
                else:
                    bins = "auto"
                hist, bin_edges = np.histogram(data, bins=bins, density=True)
                other_distributions[column] = DistributionData(hist, bin_edges, None)
            elif data_type == "categorical":
                counts_obj = data.value_counts()
                if (
                    existing_other_distributions is not None
                    and column in existing_other_distributions
                ):
                    categories = existing_other_distributions[column].categories
                else:
                    categories = list(counts_obj.index)
                count = np.asarray(
                    [counts_obj[cat] if cat in counts_obj else 0 for cat in categories]
                )
                distribution = count / np.sum(count)
                other_distributions[column] = DistributionData(
                    distribution, None, categories
                )
            else:
                raise ValueError(f"Not handled {df[column].dtype}!")

    return label_distribution, other_distributions


def split_dataset(
    df: pd.DataFrame,
    label_column: str,
    speaker_column: str,
    num_folds: Optional[int] = None,
    test_size: Optional[float] = None,
    label_distribution: Optional[DistributionData] = None,
    other_distributions: Optional[dict[str, DistributionData]] = None,
    previous_train_folds: Optional[pd.DataFrame | list[pd.DataFrame]] = None,
    previous_test_folds: Optional[pd.DataFrame | list[pd.DataFrame]] = None,
    filename_column: str = "filename",
    random_state: Optional[int] = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Split a dataset into train and test folds

    Split a dataset into train and test folds, making K non-overlapping folds or 1 fold
    based on the proportion specified by test_size. If both K-fold splitting and
    proportion-based splitting are specified, then the K-fold splitting takes
    precedence. The splits will be speaker-disjoint. If label or other distributions
    are provided, then the splits will try to match the provided distributions, based
    on measuring the KL divergence.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame containing the dataset.
    label_column : str
        Column in the DataFrame corresponding to the target labels (eg. "bin_phq").
    speaker_column : str
        Column in the DataFrame corresponding to speaker/participant ID (eg. "user").
    num_folds : int, optional
        Number of folds to create.
    test_size : float, optional
        Proportion of the dataset to allocate to the test split. Must be between 0.0
        and 1.0 (exclusive). 1 - test_size will be allocated to the train split.
    label_distribution : DistributionData, optional
        Contains distribution information about the labels. If provided, the splits
        will try to match this label distribution.
    other_distributions : dict[str, DistributionData], optional
        Contains distribution information about various columns in the DataFrame, where
        the dictionary key specifies the column name. If provided, the splits will try
        to match these distributions.
    previous_train_folds : pandas.DataFrame or list of pandas.DataFrame, optional
        Single DataFrame or list of DataFrames containing all folds of the train split
        from a previous dataset.
    previous_test_folds : pandas.DataFrame or list of pandas.DataFrame, optional
        Single DataFrame or list of DataFrames containing all folds of the test split
        from a previous dataset.
    filename_column: str, optional
        Column used to uniquely identify files in the input dataset and previous
        train/test folds.
    random_state : int, optional
        If provided, it seeds the random number generator.

    Returns
    -------
    train_folds : list[pandas.DataFrame]
        num_folds-length list containing all folds of the train split. For a
        proportion-based split, this will be a 1-length list.
    test_folds : list[pandas.DataFrame]
        num_folds-length list containing all folds of the test split. For a
        proportion-based split, this will be a 1-length list.
    """
    # Determine if creating K-fold splits are a single proportion-based split.
    if num_folds is not None:
        # Create K-fold splits. This takes precedence over proportion-based split.
        n_splits = num_folds
        split_type = "k_fold"
    elif test_size is not None:
        # Create one proportion-based split.
        n_splits = 1
        split_type = "proportion"
    else:
        raise ValueError("num_folds and test_size cannot both be None.")

    # If label or other distributions are supplied, then run multiple repetitions of
    # creating splits and choose the repetition with the lowest KL divergence to the
    # supplied distributions.
    if label_distribution is None and other_distributions is None:
        num_repetitions = 1
    else:
        num_repetitions = 10

    # Remove previous data (if specified) from the current data so that the
    # splitting operation is performed only on new additional data (relative to the
    # previous data). The previous data will be concatenated onto the new additional
    # data splits.
    # Since union of previous train and test data should be the same over all folds,
    # just need to remove previous train and test data from fold 0.
    remaining_df = df.copy(deep=True)
    if previous_train_folds is not None:
        if not isinstance(previous_train_folds, list):
            previous_train_folds = [previous_train_folds]
        # Remove samples from the previous dataset that aren't in the current data
        for idx in range(0, len(previous_train_folds)):
            train_df = previous_train_folds[idx]
            previous_train_folds[idx] = train_df[
                train_df[filename_column].isin(df[filename_column])
            ]

        # Remove samples from current data that are in the previous dataset
        remaining_df = remaining_df[
            ~remaining_df[filename_column].isin(
                previous_train_folds[0][filename_column]
            )
        ]
    if previous_test_folds is not None:
        if not isinstance(previous_test_folds, list):
            previous_test_folds = [previous_test_folds]
        # Remove samples from the previous dataset that aren't in the current data
        for idx in range(0, len(previous_test_folds)):
            test_df = previous_test_folds[idx]
            previous_test_folds[idx] = test_df[
                test_df[filename_column].isin(df[filename_column])
            ]

        # Remove samples from current data that are in the previous dataset
        remaining_df = remaining_df[
            ~remaining_df[filename_column].isin(previous_test_folds[0][filename_column])
        ]

    if len(remaining_df) == 0:
        # Current data did not have any new additional data relative to the previous
        # data, so just return the previous train/test splits.
        return previous_train_folds, previous_test_folds

    # Create data splits
    all_dataset_reps = []
    for rep in range(num_repetitions):
        seed = random_state + rep if random_state is not None else None
        if split_type == "k_fold":
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
        elif split_type == "proportion":
            splitter = GroupShuffleSplit(
                n_splits=n_splits, test_size=test_size, random_state=seed
            )
        dataset_folds = []
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(
                remaining_df.index,
                remaining_df[label_column],
                groups=remaining_df[speaker_column],
            )
        ):
            train_df = remaining_df.iloc[train_idx]
            if previous_train_folds is not None:
                train_df = pd.concat((previous_train_folds[fold], train_df))
            test_df = remaining_df.iloc[test_idx]
            if previous_test_folds is not None:
                test_df = pd.concat((previous_test_folds[fold], test_df))
            dataset_folds.append((train_df, test_df))
        all_dataset_reps.append(dataset_folds)

    # Choose a repetition with the smallest KL divergence to the supplied
    # distributions. Note that the train/test splits are paired
    # (train_dataset_splits[i], test_dataset_splits[i]), so need to choose splits
    # jointly.
    if num_repetitions == 1:
        best_dataset_rep = all_dataset_reps[0]
    else:
        if other_distributions is not None:
            other_columns = list(other_distributions.keys())
        else:
            other_columns = None

        kl_divergences = []
        for rep in range(num_repetitions):
            divergence = 0
            for train_df, test_df in all_dataset_reps[rep]:
                # Iterate through each fold in this repetition
                df_dict = dict()
                df_dict["train"] = train_df
                df_dict["test"] = test_df

                # Calculate label/other distributions for the train/test splits using
                # the same bins as the supplied distributions. Then, calculate the KL
                # divergence of the distributions summed across train and test splits.
                for data_split in ["train", "test"]:
                    label_dist, other_dist = calculate_distributions(
                        df_dict[data_split],
                        label_column,
                        other_columns=other_columns,
                        existing_label_distribution=label_distribution,
                        existing_other_distributions=other_distributions,
                    )

                    if label_distribution is not None:
                        divergence += entropy(
                            label_dist.distribution, label_distribution.distribution
                        )
                    if other_distributions is not None:
                        for column in other_columns:
                            divergence += entropy(
                                other_dist[column].distribution,
                                other_distributions[column].distribution,
                            )
            kl_divergences.append(divergence)
        sorted_dataset_reps = sorted(
            list(zip(all_dataset_reps, kl_divergences)), key=lambda x: x[1]
        )
        best_dataset_rep = sorted_dataset_reps[0][0]  # keep the best repetition

    train_folds, test_folds = zip(*best_dataset_rep)
    train_folds = list(train_folds)
    test_folds = list(test_folds)

    return train_folds, test_folds


def get_data_type(series: pd.Series) -> str:
    """Determine if a Pandas Series has numerical or categorical data.

    Parameters
    ----------
    series : pandas.Series
        Pandas Series object.

    Returns
    -------
    data_type : str
        Either "numerical" or "categorical".
    """
    try:
        # TODO: distinguish between integers for numerical and categorical data.
        pd.to_numeric(series, errors="raise")
        data_type = "numerical"
    except (ValueError, TypeError):
        data_type = "categorical"

    return data_type


def merge_df_rows_by_subject(df):
    # Move UUID index back to column to make it easier to recover the UUIDs from the
    # merged rows.
    df = df.reset_index()

    common_cols = [
        "wave",
        "user",
        "ethnicity",
        "age",
        "gender",
        "zip_code",
        "state",
        "income",
        "english_preferred",
        "preferred_lang",
        "userAgent",
        "phq",
        "phq_2",
        "gad",
        "gad_2",
        "bin_phq",
        "bin_gad",
        "source",
        "pmi_score",
    ]
    common_cols.extend([f"phq{i}" for i in range(1, 10)])
    common_cols.extend([f"gad{i}" for i in range(1, 8)])
    if "real" in df.columns:
        common_cols.append("real")
    list_cols = [
        "uuid",
        "filename",
        "orig_audio_name",
        "prompt",
        "asr_output_id",
        "asr_operation_name",
        "asr_operation_start_time",
    ]
    sum_cols = [
        "duration",
        "prompt_vad_cut_duration",
        "prompt_vad_trimmed_duration",
        "conversational_vad_cut_duration",
        "conversational_vad_trimmed_duration",
    ]
    # If KD targets are provided, output the mean of the provided targets
    mean_cols = [c for c in df.columns if c.startswith("scores")]
    common_cols = [c for c in common_cols if c in df.columns]
    list_cols = [c for c in list_cols if c in df.columns]
    sum_cols = [c for c in sum_cols if c in df.columns]
    if "prompt" in df.columns:
        df["prompt"] = df["prompt"].fillna("")
    groupby = df.groupby(common_cols, dropna=False)
    agg = groupby[sum_cols].sum()
    agg = agg.join(groupby[mean_cols].mean())
    for list_col in list_cols:
        agg = agg.join(groupby[list_col].apply(list))

    # Make UUID as row index. Since UUIDs have been merged into a list, take the first
    # UUID in the list as the UUID index for the row after sorting the UUIDs.
    agg = agg.reset_index()
    agg["uuid"] = agg["uuid"].apply(lambda x: sorted(x)[0])
    agg.set_index("uuid", inplace=True)

    return agg


def repeat_along_batch_dim(
    data: torch.Tensor | Mapping[str, torch.Tensor], repetitions: int = 1
) -> torch.Tensor | Mapping[str, torch.Tensor]:
    if isinstance(data, torch.Tensor):
        repeats = [repetitions] + [1 for _ in range(0, data.dim() - 1)]
        return data.repeat(*repeats)
    elif isinstance(data, Mapping):
        return {
            key: repeat_along_batch_dim(val, repetitions=repetitions)
            for key, val in data.items()
        }
    else:
        raise ValueError(f"Unable to handle data of type {type(data)}.")


def broadcast_along_batch_dim(data: Any) -> tuple[Any, int]:
    """Broadcast tensors in a nested structure to have the same batch dim, returning
    the resulting structure and dim.

    Parameters
    ----------
    data : torch.Tensor or data structure containing torch.Tensor
        Tensor or data structure (eg. dictionary or list) containg tensors to broadcast
        along the batch dimension. When multiple tensors are provided, the batch
        dimension of each tensor is broadcast to match the largest batch dimension
        across the tensors.

    Returns
    -------
    broadcast_data : torch.Tensor or data structure containing torch.Tensor
        Data in the same structure as the input but with the broadcasted batch
        dimensions for each tensor contained.
    length : int
        Size of the batch dimension that each tensor was broadcast to.
    """

    def max_batch_dim(data) -> int:
        if isinstance(data, torch.Tensor):
            return data.shape[0]
        if isinstance(data, Mapping):
            return max_batch_dim(data.values())
        if isinstance(data, Iterable):
            return max(max_batch_dim(d) for d in data)
        raise ValueError(
            f"Got {data=} with invalid {type(data)=} that is not Tensor, Mapping, or "
            "Iterable."
        )

    def broadcast_to(data, length):
        if isinstance(data, torch.Tensor):
            batch_dim = data.shape[0]
            repetitions, remainder = divmod(length, batch_dim)
            if remainder:
                raise ValueError(
                    "Cannot broadcast batch dimension of tensor with shape "
                    f"{data.shape} to {length}."
                )
            return repeat_along_batch_dim(data, repetitions=repetitions)
        if isinstance(data, Mapping):
            return {key: broadcast_to(val, length) for key, val in data.items()}
        if isinstance(data, Iterable):
            return [broadcast_to(d, length) for d in data]

    length = max_batch_dim(data)
    return broadcast_to(data, length), length


def get_collate_fn(tensor_collate_fn: Callable[[list[torch.Tensor]], torch.Tensor]):
    def tensor_collate_tensor_fn(
        batch, *args, **kwargs
    ):  # torch internals pass some args we don't need
        return tensor_collate_fn(batch)

    collate_fn_map = copy.copy(default_collate_fn_map)
    collate_fn_map[torch.Tensor] = tensor_collate_tensor_fn
    return partial(collate, collate_fn_map=collate_fn_map)


concat_collate = get_collate_fn(torch.concat)
concat_collate_to_cpu = get_collate_fn(lambda batch: torch.concat(batch).to("cpu"))


def broadcast_collate(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate function that broadcasts the batch dimension of the features for each
    modality to match the largest batch dimension across the modalities.

    Parameters
    ----------
    batch : list of dictionaries
        batch[i] contains a dictionary that maps "metadata" to a dictionary containing
        a UUID and label, and modality (eg. "audio" or "text") that maps to features.

    Returns
    -------
    Dictionary containing UUIDs, features, lengths of the features, and labels of each
    sample.
    """

    def broadcast_desired(batch):
        out = {
            **batch["metadata"]
        }  # contains DatasetFields.UUID and DatasetFields.LABEL
        modality_data = {key: val for key, val in batch.items() if key != "metadata"}
        modality_data_broadcast, length = broadcast_along_batch_dim(modality_data)
        out[DatasetFields.LENGTH] = length
        out[DatasetFields.FEATURES] = modality_data_broadcast

        return out

    return concat_collate(list(map(broadcast_desired, batch)))
