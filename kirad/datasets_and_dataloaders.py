"""Base datasets and data loaders. Currently only contains the audio loader for
the Whisper Medium backbone."""

import operator
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import soundfile
import torch
import torchaudio
from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr
from torch.utils.data import Dataset, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)

from .constants import (
    EXPECTED_SAMPLE_RATE,
    DatasetFields,
    DatasetSplit,
    DatasetVersion,
    DataSource,
    FeatureStoreBucket,
)
from .dataset_utils import (
    load_relative_dataset,
    merge_df_rows_by_subject,
    resolve_audio_path,
)
from .utils import (
    get_wandb_run_name_tag_from_artifact_url,
    import_object,
    localize_wandb_artifact,
    read_google_asr_json,
    read_whisper_timestamped_json,
)

INFERENCE_WINDOW_SIZE = 30
MAX_INFERENCE_WINDOW_OVERLAP_FRACTION = 1 / 3
INFERENCE_BATCH_SIZE = 16

DEFAULT_LABEL_COLUMN = {"depression": "phq", "anxiety": "gad"}
MERGE_ROWS_BY_SUBJECT_VALUES = Literal[False, "many_to_one", "many_to_many"]
TEXT_OUTPUT_PER_SAMPLE_VALUES = Literal["cyclic", "merge"]
WINDOW_METHOD_VALUES = Literal[
    "all", "single_random", "single_start", "random_adjacent_pair"
]


class KintsugiMetadata(BaseModel, Dataset):
    """
    Class for reading and processing metadata for training.

    Attributes
    ----------
    df_or_dataset_version : pd.DataFrame or string
        If DataFrame, must contain the filename of the files being loaded in
        the `filename` column. If a string, specifies the dataset version to
        load from Kintsugi's standardized dataset splits.
    audio_root : Path or string
        If a Path, defines the directory from which audio files should be
        loaded. If a string, defines the feature store bucket to load audio
        from. Available feature store bucket options are `raw`, `resampled`,
        `resampled_and_cut_audio_with_conversational_vad_params`,
        `resampled_and_trimmed_audio_with_conversational_vad_params`,
        `resampled_and_cut_audio_with_prompt_vad_params`, and
        `resampled_and_trimmed_audio_with_prompt_vad_params`.
    text_root : Path or string
        Path to directory containing transcription json files.
    data_sources : string or list of strings, optional
        If `df_or_dataset_version` defines a dataset version, the
        `data_sources` parameter defines which data sources to load. For
        example, if `data_sources` is `phonic`, only the `phonic` metadata
        is used in the specified Kintsugi dataset. Available data sources are
        `phonic`, `sonar`, `sonar_general_pop`, `sonar_senior_pop`,
        `sonar_elderly_and_latino_pop`, `fda`, `app`, and `all`. The `sonar` data
        source only exists in the v1 dataset and refers to the senior population data.
        The data from Sonar was broken into `senior_general_pop` and
        `sonar_senior_pop` in later datasets.
    dataset_splits : string or list of strings, optional
        If `df_or_dataset_version` defines a dataset version, the
        `dataset_split` parameter defines which split to load. For example,
        if `dataset_split` is `train`, only the train split is loaded. Available
        splits are `train`, `val`, `test`, and `all`
    dataset_fold : int, optional
        If `df_or_dataset_version` defines a dataset version, the
        `dataset_fold` parameter defines which fold to load.
    label_column : dict, optional
        Specify a dictionary with (task name, label_column) key/value pairs for the
        column containing the ground truth scores for each task
    duration_column : string, optional
        Specify column containing the audio duration.
    merge_rows_by_subject : Literal[False, "many_to_one", "many_to_many"]
        If False, leave data as is, without combining rows corresponding to the same subject in any
        way -- this should always be the choice for val and test.

        Otherwise, replace all the rows corresponding to a subject with either a single row
        ("many_to_one") or the same number of rows ("many_to_many"), as follows. In both cases the
        audio is concatenated across all these rows before windows / features are extracted. In the
        "many_to_one" case the text output is the text of a random one of the subjects' rows,
        chosen independently at each epoch. In the "many_to_many" case the text isn't changed -- if
        a subject had three streams before merging with texts A, B, and C, the dataset will still
        contain three streams, one with each of texts A, B, and C. Only the audio is concatenated,
        so it no longer corresponds directly to the text.

        The way these options interact with `window_method` to affect text output is slightly
        counterintuitive, but is chosen to make score variance loss work effectively. In case
        `merge_rows_by_subject` is not False and `window_method` is set to return multiple windows,
        then the text for the first window is chosen as described above and the text for subsequent
        windows cycle through other text from that subject. Continuing the above three-stream
        example, if the text output for the first window would have been B, the text output for the
        second window would be C, for the third window it would be A, for the fourth B, etc.
    window_duration : int
        The number of seconds to grab from each audio file before feature extraction.
    max_overlap_frac : float, optional
        Specify how much the windows overlap as a fraction of the window length.
        Default: 0.0.
    score_targets_csv : Optional[str]
        If present, filename of a CSV file with columns "uuid" and f"scores_{task}" for each task in
        `label_type`. These scores will be packaged with the quantized labels for the elements of
        this dataset, so they can be used by the model e.g. as targets for knowledge distillation.
        This CSV is typically generated by:
          (1) Running evaluation of a model using `eval_set = "train"`.
          (2) Identifying the WANDB_RUN_ID of this eval job
          (3) Running `kirad.utils.get_dataframe_with_scores(WANDB_RUN_ID, "test").to_csv(OUT_CSV)`
              (note this must be called with split="test", not split="train" -- see the comment
              in `get_dataframe_with_scores` for a description of this inconsistency.)
        but could alternatively be generated by some other ad hoc process, e.g. combining scores
        of multiple models in some way.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # Specify input
    df_or_dataset_version: pd.DataFrame | DatasetVersion | str | Path | Mapping | list[
        pd.DataFrame | DatasetVersion | str | Path | Mapping
    ]
    data_sources: Optional[DataSource | list[DataSource]] = None
    dataset_splits: Optional[DatasetSplit | list[DatasetSplit]] = None
    dataset_fold: Optional[int] = None
    # Specify columns in the metadata
    label_column: Mapping[str, str] = DEFAULT_LABEL_COLUMN
    duration_column: Optional[str] = None
    # Preprocessing options
    merge_rows_by_subject: MERGE_ROWS_BY_SUBJECT_VALUES = False
    window_duration: int = INFERENCE_WINDOW_SIZE
    max_overlap_frac: float = 0.0
    # Filtering options
    fraction: float = 1.0
    min_duration: Optional[float] = INFERENCE_WINDOW_SIZE
    max_duration: Optional[float] = None
    # Label creation options
    score_targets_csv: Optional[str | Path] = None
    audio_root: Optional[str | Path] = None
    text_root: Optional[str | Path] = None

    # Private attributes that are not a part of the config
    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.df_or_dataset_version, list):
            self.df_or_dataset_version = [self.df_or_dataset_version]
        self._df = pd.concat(
            load_relative_dataset(
                item,
                self.data_sources,
                self.dataset_splits,
                self.dataset_fold,
                audio_root=self.audio_root,
                text_root=self.text_root,
            )
            for item in self.df_or_dataset_version
        )
        # Index rows (samples) by UUID
        self._df.set_index("uuid", inplace=True)
        self.preprocess_metadata()
        self.filter_metadata()

    def preprocess_metadata(self):
        if self.score_targets_csv is not None:
            score_target_df = pd.read_csv(self.score_targets_csv)
            score_target_df.set_index("uuid", inplace=True)
            self._df = self._df.merge(
                score_target_df,
                how="left",
                left_index=True,
                right_index=True,
                suffixes=(None, "_target"),
            )

        if self.merge_rows_by_subject:
            self._df = merge_df_rows_by_subject(self._df)

            if self.merge_rows_by_subject == "many_to_many":
                self._df["num_files"] = self._df["filename"].map(
                    lambda fn: 1 if isinstance(fn, str) else len(fn)
                )
                max_files = self._df["num_files"].max()
                window_indices = pd.DataFrame(
                    [(i, j) for i in range(1, max_files + 1) for j in range(i)],
                    columns=["num_files", "text_index"],
                )
                self._df = self._df.merge(window_indices, on="num_files", how="left")

        # Calculate the maximum number of audio windows per sample based on window
        # duration and window overlap fraction.
        self._df["num_windows"] = self._df[self.duration_column].map(
            lambda d: len(
                get_window_starts(
                    "all",
                    int(d * EXPECTED_SAMPLE_RATE),
                    self.window_duration * EXPECTED_SAMPLE_RATE,
                    self.max_overlap_frac,
                )
            )
        )

    def filter_metadata(self):
        self._df = self._df.sample(frac=self.fraction, random_state=1)
        logger.info(f"Keeping {self.fraction:0.2%} of samples.")
        orig_len = len(self._df)
        mask = pd.Series([True for _ in range(0, orig_len)], index=self._df.index)

        # Filter out files based on audio duration.
        for duration, op_func, logging_str in [
            (self.min_duration, operator.ge, "less than"),
            (self.max_duration, operator.le, "greater than"),
        ]:
            if duration is not None:
                if self.duration_column is None:
                    raise ValueError(
                        "`duration_column` must be specified to filter metadata by "
                        "duration."
                    )
                duration_mask = op_func(self._df[self.duration_column], duration)
                logger.info(
                    f"Filtered out {(~duration_mask).sum()} files that were "
                    f"{logging_str} {duration} seconds."
                )
                mask &= duration_mask

        # Apply all filters
        self._df = self._df[mask]
        logger.info(
            f"""
            Original number of files\t: {orig_len}
            Remaining number of files\t: {mask.sum()}
            Number of files filtered\t: {(~mask).sum()}
            """
        )

    def __getitem__(self, uuid: str) -> dict[str, str | dict[str, int | float]]:
        row = self._df.loc[uuid]
        label = {
            task: {DatasetFields.LabelFields.PRIMARY: row[label_column]}
            for task, label_column in self.label_column.items()
        }

        # Add KD targets if provided
        for task in self.label_column:
            if f"scores_{task}" in row:
                label[task][DatasetFields.LabelFields.KD] = row[f"scores_{task}"]

        return {DatasetFields.UUID: uuid, DatasetFields.LABEL: label}

    def __len__(self) -> int:
        return len(self._df)


class KintsugiDatasetBase(
    BaseModel,
    Dataset,
):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    metadata: KintsugiMetadata

    def __len__(self) -> int:
        return len(self.metadata)


def dummy_preprocessor(x, *args, **kwargs):
    return torch.from_numpy(np.array(x))


class KintsugiAudioDataset(KintsugiDatasetBase):
    """
    An audio dataset that processes and generates log-mel spectrogram features from
    audio files.

    Attributes
    ----------
    metadata : KintsugiMetadata
        Metadata object
    preprocessor : string
        Which preprocessor to use. Allowable values can be found at

            https://huggingface.co/models?search=openai/whisper

        and include "openai/whisper-large", "openai/whisper-medium.en",
        "openai/whisper-small.en" (default), or "openai/whisper-base.en".
    normalize_audio : bool, optional
        If true, then scale the audio waveform to be between -1 and 1.
    normalize_features : bool, optional
        If true, then normalize the mel-spectrogram to have zero-mean,
        unit-variance.
    window_duration : int
        The number of seconds to grab from each audio file before feature
        extraction
    window_method : str, optional
        Specify how that audio is windowed prior to extracting features. The window
        length is specified by `window_duration`. Can be "all" (default),
        "single_random", "single_start", or "random_adjacent_pair".
    max_overlap_frac : float, optional
        Specify how much the windows overlap as a fraction of the window length.
        Default: 0.0.
    ideal_logmel_energies : list or numpy.ndarray, optional
        Specifies the value of each log-mel bin to normalize to. The length of the
        array must match the number of log-mel bins (eg. 80 for
        WhisperFeatureExtractor).
    """

    preprocessor: str | Literal[False] = "openai/whisper-small.en"
    normalize_audio: bool = False
    normalize_features: bool = False
    window_duration: int = INFERENCE_WINDOW_SIZE
    window_method: WINDOW_METHOD_VALUES = "all"
    max_overlap_frac: float = 0.0
    ideal_logmel_energies: Optional[npt.ArrayLike | str | Path] = None
    pad_last_chunk_to_full: bool = False
    augmentations: Sequence[Mapping] = ()

    # Private attributes that are not a part of the config
    _preprocessor_with_audio_normalization: Callable[..., torch.Tensor] = PrivateAttr()
    _augmentations: list[Tuple[float, Callable]] = PrivateAttr(default_factory=list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ideal_logmel_energies is not None:
            if isinstance(self.ideal_logmel_energies, str | Path):
                self.ideal_logmel_energies = np.load(self.ideal_logmel_energies)
            self.ideal_logmel_energies = torch.tensor(self.ideal_logmel_energies)

        #
        for augmentation_config in self.augmentations:
            class_name = augmentation_config["class_name"]
            params = augmentation_config["params"]
            aug_prob = augmentation_config["aug_prob"]

            aug_factory = import_object(class_name)
            augmentation = aug_factory(**params)

            self._augmentations.append((aug_prob, augmentation))

        self.init_data_processor()

    def init_data_processor(self):
        if self.preprocessor:
            preprocessor = AutoFeatureExtractor.from_pretrained(self.preprocessor)
        else:
            preprocessor = dummy_preprocessor
        self._preprocessor_with_audio_normalization = (
            get_preprocessor_with_audio_normalization(
                preprocessor=preprocessor,
                window_method=self.window_method,
                normalize_audio=self.normalize_audio,
                normalize_features=self.normalize_features,
                trim_duration=self.window_duration,
                ideal_logmel_energies=self.ideal_logmel_energies,
                max_overlap_frac=self.max_overlap_frac,
                pad_last_chunk_to_full=self.pad_last_chunk_to_full,
            )
        )

    def __getitem__(self, uuid: str) -> torch.Tensor:
        """Loads an audio tensor.

        Parameters
        ----------
        uuid : str
            The UUID of the sample.

        Returns
        -------
        Tensor containing audio features.

        Raises
        ------
        RuntimeError
            If the sample rate of the audio file to be loaded is not
            `TARGET_SAMPLE_RATE`
        """
        row = self.metadata._df.loc[uuid]
        filenames = (
            row.audio_filename
            if isinstance(row.audio_filename, list)
            else [row.audio_filename]
        )
        audio = torch.zeros(0)
        srs = set()
        for fp in filenames:
            try:
                new_audio, sr = torchaudio.load(fp)
            except soundfile.LibsndfileError as e:
                raise ValueError(f"Failed to load audio from {fp}") from e

            if sr != EXPECTED_SAMPLE_RATE:
                logger.warning(
                    f"Found sample rate of {sr} for file {fp}, which doesn't match "
                    f"the expected sample rate of {EXPECTED_SAMPLE_RATE}."
                )

            if new_audio.dim() == 2:
                new_audio = new_audio[0]
            audio = torch.cat((audio, new_audio), dim=0)
            srs.add(sr)

        if len(srs) != 1:
            raise ValueError(
                "Cannot concatenate audios with different sample rates: found sample "
                f"rates {srs} among file names {filenames}."
            )

        # applying augmentations
        for aug_prob, aug in self._augmentations:
            if random.random() < aug_prob:
                audio = aug(audio)

        features = self._preprocessor_with_audio_normalization(audio, sr)
        if "window_index" in self.metadata._df.columns:
            window_index_a = row.window_index
            window_index_b = (window_index_a + 1) % row.num_windows
            features = torch.stack(
                [features[window_index_a, ...], features[window_index_b, ...]], dim=0
            )

        return features


class KintsugiTextDataset(KintsugiDatasetBase):
    """
    A text dataset that processes and tokenizes strings using a tokenizer (eg. BERT
    tokenizer).

    Attributes
    ----------
    metadata : KintsugiMetadata
        Metadata object
    preprocessor : str
        Which Tokenizer preprocessor to use. Can be "google-bert/bert-base-uncased".  See

            https://huggingface.co/models?search=google-bert/bert

        for other possible values.
    prompt : str
        Prompt for generative LLM model
    max_tokens : int
        Max sequence length to which the sequence is padded or truncated at.
        If not provided, the default is used.
    text_output_per_sample : str
        Specifies how to handle multiple text strings per sample (eg. due to setting
        `merge_rows_by_subject` in `KintsugiMetadata`).
        - "cyclic": starting from a specified index in the "text_index" column or a
          random index if "text_index" is missing, take num_audio_window strings from
          the string list (wrap around as needed, hence the name "cyclic")
        - "merge": merge all strings in the list into a single string.
    """

    preprocessor: str | Path = "google-bert/bert-base-uncased"
    text_output_per_sample: TEXT_OUTPUT_PER_SAMPLE_VALUES = "merge"
    prompt: Optional[str] = None
    max_tokens: Optional[int] = None

    # Private attributes that are not a part of the config.
    _tokenizer: Callable[[str], BatchEncoding] = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_data_processor()

    def init_data_processor(self):

        tokenizer_path = self.preprocessor
        if self.preprocessor.startswith("https:"):
            run, name, tag = get_wandb_run_name_tag_from_artifact_url(self.preprocessor)
            tokenizer_path = localize_wandb_artifact(run, name, tag, return_dir=True)

        # use_fast=False prefers regular tokenizer with full functionality
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self._tokenizer = get_tokenizer(tokenizer, self.prompt, self.max_tokens)

    def __getitem__(self, uuid: str) -> dict[str, torch.Tensor]:
        """Loads an audio tensor.

        Parameters
        ----------
        uuid : str
            The UUID of the sample.

        Returns
        -------
        Dictionary containing text features.
        """
        row = self.metadata._df.loc[uuid]

        # Experiments at https://app.clickup.com/20576956/v/dc/kkynw-19251/kkynw-112271
        # showed that whisper large transcripts perform better than google transcripts
        filenames = (
            row.text_filename
            if isinstance(row.text_filename, list)
            else [row.text_filename]
        )
        asr_output_ids = (
            row.asr_output_id
            if isinstance(row.asr_output_id, list)
            else [row.asr_output_id]
        )
        texts = []
        for json_filename, asr_output_id in zip(filenames, asr_output_ids):
            try:
                texts.append(read_whisper_timestamped_json(json_filename))
            except FileNotFoundError:
                texts.append(read_google_asr_json(json_filename))

        if self.text_output_per_sample == "cyclic":
            num_audio_windows = row["num_windows"]
            text_index = row.get("text_index", random.randint(0, len(texts) - 1))
            text_output = [
                texts[(text_index + i) % len(texts)] for i in range(num_audio_windows)
            ]
        elif self.text_output_per_sample == "merge":
            text_output = " ".join(texts)

        features = self._tokenizer(text_output)

        return features


class KintsugiCacheDataset(KintsugiDatasetBase):
    """A dataset that returns features from a backbone cache.

    Attributes
    ----------
    metadata : KintsugiMetadata
        Metadata object
    backbone_cache : pandas DataFrame with 2 columns [uuid, feat]

    Sometimes(e.g. LLMA training case) we can be interested in a specific feature slice of the
    feature vectors in the backbone cache.
    feat_start_idx and feat_end_idx params define start and end indices of such a slice.
    feat_end_idx is a non-inclusive index.
    """

    backbone_cache: pd.DataFrame | str | Path
    feat_start_idx: Optional[int] = 0
    feat_end_idx: Optional[int] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.backbone_cache, str | Path):
            # Load cache from W&B
            cache_run, cache_name, cache_tag = get_wandb_run_name_tag_from_artifact_url(
                self.backbone_cache
            )
            cache_path = localize_wandb_artifact(cache_run, cache_name, cache_tag)
            state_dict = torch.load(cache_path)
            self.backbone_cache = pd.DataFrame(
                state_dict["backbone_cache"].items(), columns=["uuid", "feat"]
            )
        if not {"uuid", "feat"}.issubset(set(self.backbone_cache.columns)):
            raise ValueError(
                "Backbone cache data frame should have columns 'uuid' and 'feat'."
            )

        # Index the cache by UUIDs
        if (
            self.backbone_cache.index.name is None
            or self.backbone_cache.index.name != "uuid"
        ):
            if self.backbone_cache.index.name != "uuid":
                self.backbone_cache.reset_index(inplace=True)
            self.backbone_cache.set_index("uuid", inplace=True)

        self._process_cache()

    def _process_cache(self):

        # Filter out UUIDs from the metadata that aren't in the backbone cache.
        mask = self.metadata._df.index.isin(self.backbone_cache.index)
        logger.info(
            f"Filtered out {(~mask).sum()} files that are not in the backbone cache."
        )
        if self.metadata.fraction < 1:
            logger.info(
                f"Warning: 'fraction' parameter applies to the original metadata, "
                f"and it may have an approximate effect with respect to the cache content."
            )
        self.metadata._df = self.metadata._df[mask]

    def __getitem__(self, uuid: str) -> torch.Tensor:
        # we assume the features are contained in dim = 1, but tensor can have
        # an arbitrary number of dimensions
        return self.backbone_cache.loc[uuid]["feat"][
            :, self.feat_start_idx : self.feat_end_idx, ...
        ]


class UuidSampler(Sampler[str]):
    """Provides a random sampler to sample UUIDs from the metadata."""

    def __init__(
        self,
        metadata: KintsugiMetadata,
        shuffle: bool = False,
        distributed: bool = False,
    ):
        self.uuids = metadata._df.index.tolist()
        if distributed:
            self.sampler = DistributedSampler(self.uuids, shuffle=shuffle)
        else:
            if shuffle:
                self.sampler = RandomSampler(
                    self.uuids, replacement=False, num_samples=None, generator=None
                )
            else:
                self.sampler = SequentialSampler(self.uuids)

    def __iter__(self) -> Iterator[str]:
        for idx in self.sampler:
            yield self.uuids[idx]

    def __len__(self) -> int:
        return len(self.uuids)


def get_window_starts(
    method: WINDOW_METHOD_VALUES,
    audio_len: int,
    inference_window_samples: int,
    max_inference_window_overlap_fraction: float = MAX_INFERENCE_WINDOW_OVERLAP_FRACTION,
) -> int | Iterable[int]:
    """Calculate the audio windows based on the audio duration. Each audio window is inference_window_samples long.

    A maximum overlap of max_inference_window_overlap_samples is used, but the overlap is flexible so that as much audio
    as possible is used.

    Arguments
    ---------
        method : The method used for selecting window starts
        audio_len : the length of the audio in samples
        inference_window_samples : the length of the inference window in samples
        max_inference_window_overlap_fraction : fraction of each inference window allowed to overlap with the previous
            one (between 0.0 and 1.0)

    Returns
    -------
        For method == "all", a list of window starts overlapping by at most the specified fraction.
        For method == "single_random", a single window start at a random position.

    """
    overflow_len = audio_len - inference_window_samples
    if method == "single_start":
        return 0
    if method == "single_random":
        return random.randint(0, max(overflow_len, 0))
    if method == "random_adjacent_pair":
        if audio_len < 2 * inference_window_samples:  # overlap unavoidable
            return [0, overflow_len]  # minimize overlap
        first = random.randint(0, audio_len - 2 * inference_window_samples)
        return [first, first + inference_window_samples]
    if method == "all":
        min_hop_samples = int(
            (1 - max_inference_window_overlap_fraction) * inference_window_samples
        )

        n_windows = 1 + overflow_len // min_hop_samples
        return np.linspace(0, overflow_len, max(n_windows, 1)).astype(int)
    raise ValueError(f"Unknown window method: {method}")


def get_preprocessor_with_audio_normalization(
    preprocessor: Callable,
    window_method: WINDOW_METHOD_VALUES,
    normalize_audio: bool = False,
    normalize_features: bool = False,
    trim_duration: int = INFERENCE_WINDOW_SIZE,
    ideal_logmel_energies: Optional[torch.FloatTensor] = None,
    max_overlap_frac: float = MAX_INFERENCE_WINDOW_OVERLAP_FRACTION,
    pad_last_chunk_to_full: bool = False,
) -> Callable[..., torch.Tensor]:
    def forward(
        audio: torch.Tensor, sampling_rate: int = EXPECTED_SAMPLE_RATE
    ) -> torch.Tensor:
        if normalize_audio:
            # Remove DC offset and scale amplitude to [-1, 1]
            audio = audio - torch.mean(audio)
            audio = audio / torch.max(torch.abs(audio))

        chunk_samples = sampling_rate * trim_duration

        # pad audio in a way, so that the last chunk is not dropped
        if pad_last_chunk_to_full:
            if max_overlap_frac > 0:
                raise ValueError(
                    f"pad_last_chunk_to_full is only supported for non-overlapping windows"
                )
            num_chunks = np.ceil(len(audio) / chunk_samples)
            pad_size = int(num_chunks * chunk_samples - len(audio))
            audio = torch.nn.functional.pad(audio, (0, pad_size))

        window_starts = get_window_starts(
            window_method, len(audio), chunk_samples, max_overlap_frac
        )

        if isinstance(window_starts, Iterable):
            all_window_starts = window_starts
        else:
            all_window_starts = [window_starts]
        features = preprocessor(
            [
                audio[start : start + chunk_samples].numpy(force=True)
                for start in all_window_starts
            ],
            return_tensors="pt",
            sampling_rate=sampling_rate,
            do_normalize=normalize_features,
        )
        for key in ("input_features", "input_values"):
            if hasattr(features, key):
                features = getattr(features, key)
                break

        if ideal_logmel_energies is not None:
            mean_features = torch.mean(features, dim=-1)
            # features are [batch, n_logmel_bins, n_frames]
            rescale_factor = ideal_logmel_energies.unsqueeze(0) - mean_features
            rescale_factor = rescale_factor.unsqueeze(2)
            features += rescale_factor
        return features

    return forward


def get_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    prompt: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Callable[[str], BatchEncoding]:
    """Get a callable function that tokenizes text with a specified tokenizer.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        The tokenizer to invoke (eg. BertTokenizer.from_pretrained("bert-base-uncased"))

    prompt : str
        Text which prompts generative LLM model. If provided, assume we are dealing with generative LLM.

    max_length: int
        Max sequence length. If provided, it will override default max sequence length.

    Returns
    -------
    Callable
        Function that takes text and returns the tokenization in a dictionary
        containing the following:
            - "input_ids": token IDs corresponding to the tokenization of the text
            - "token_type_ids": token type IDs
            - "attention_mask": specifies which tokens the LLM should attend to
    """

    # Not every tokenizer has pad token assigned by default.
    # For some (such as Llama) it should be specified manually.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def forward(text: str) -> BatchEncoding:
        # If a prompt is specified assume we are dealing with generative LLM model
        if prompt is not None:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Some tokenizers don't have "prepare_for_tokenization" method
        if hasattr(tokenizer, "prepare_for_tokenization"):
            text, _ = tokenizer.prepare_for_tokenization(text)

        tokenized = tokenizer(
            text,
            add_special_tokens=True,
            max_length=tokenizer.model_max_length if max_length is None else max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return tokenized

    return forward
