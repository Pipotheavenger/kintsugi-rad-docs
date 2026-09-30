"""Global constants: sample rate, score ranges, label cutoffs, batch field names.

Stage: utils (used by data, model, loss and metrics). Cutoffs and MAX_SCORE define
the ordinal tasks; change them if a new questionnaire or threshold is used.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch

# HARDCODED: audio must already be 16 kHz; files are not resampled.
EXPECTED_SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
MAX_FINITE_FLOAT32 = float(np.finfo(np.float32).max)

# Defines VAD parameters for different types of audio
# HARDCODED: VAD silence/speech minimums (ms); VAD itself runs upstream, not in this repo.
DEFAULT_VAD_MIN_SILENCE_DURATION_MS = 100
DEFAULT_VAD_MIN_SPEECH_DURATION_MS = 250
VAD_PARAMS = {
    "prompt": {"min_silence_duration_ms": 100, "min_speech_duration_ms": 250},
    "conversational": {"min_silence_duration_ms": 1000, "min_speech_duration_ms": 5000},
    "inference": {
        "min_silence_duration_ms": DEFAULT_VAD_MIN_SILENCE_DURATION_MS,
        "min_speech_duration_ms": DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
    },
}

# Define maximum scores for depression (PHQ-9, PHQ-2, and SCID) and anxiety (GAD-7 and
# GAD-2).
# HARDCODED: questionnaire maxima; set the ordinal range of each task.
MAX_SCORE = {"phq9": 27, "phq2": 6, "scid": 1, "gad7": 21, "gad2": 6}

# PHQ-9 (low/medium cutoff, medium/high cutoff) = (9, 14)
# HARDCODED: binary cuts for metrics/thresholds; label counts cutoffs exceeded (strict >).
METRIC_TARGET_CUTOFFS = {"depression": (9, 14), "anxiety": (4, 9, 14), "real": (0,)}
# HARDCODED: CORAL target sets per task; exact28/exact22 use every score boundary.
LOSS_TARGET_CUTOFFS = {
    "real": {
        "exact2": (0,),
    },
    "depression": {
        "LvMH": (METRIC_TARGET_CUTOFFS["depression"][0],),  # Predict L vs. M+H
        "LMvH": (METRIC_TARGET_CUTOFFS["depression"][1],),  # Predict L+M  vs. H
        "LvMvH": METRIC_TARGET_CUTOFFS["depression"],  # Predict L vs. M vs. H
        "exact28": range(MAX_SCORE["phq9"]),  # Predict PHQ-9 score exactly
    },
    "anxiety": {
        "{2,3}_vs_{0,1}": (
            METRIC_TARGET_CUTOFFS["anxiety"][1],
        ),  # Predict moderately severe + severe vs. minimal + moderate
        "{0}_vs_{1}_vs_{2}_vs_{3}": METRIC_TARGET_CUTOFFS[
            "anxiety"
        ],  # Predict minimal vs. moderate vs. moderately severe vs. severe
        "exact22": range(MAX_SCORE["gad7"]),  # Predict GAD-7 score exactly
    },
}


# Defines dataset types and metadata for datasets
DatasetVersion = Literal[
    "v1",
    "v1.0.1",
    "v2.0",
    "v2.0.1",
    "v2.1",
    "v2.1.1",
    "v2.2",
    "v2.2.1",
    "v2.3.1",
    "v2.4.1",
    "v2.5.1",
    "v2.6.1",
    "v2.7.1",
    "v2.8.1",
    "v2.9.1",
    "v2.9.2",
    "v2.9.3",
    "v2.9.4",
    "v2.9.5",
    "v2.10.1",
    "v2.10.4",
    "v2.11.1",
    "v2.11.4",
]
# The `sonar` data source only exists in dataset v1 and refers to the senior population data.
# The `sonar` data source was broken into `sonar_general_pop` and `sonar_senior_pop` in later datasets.
DataSource = Literal[
    "phonic",
    "sonar",
    "sonar_general_pop",
    "sonar_senior_pop",
    "sonar_elderly_and_latino_pop",
    "general_pop",
    "fda",
    "app",
    "all",
]
DatasetSplit = Literal["train", "val", "test", "all"]
FeatureStoreBucket = Literal[
    "raw",
    "resampled",
    "resampled_and_cut_audio_with_conversational_vad_params",
    "resampled_and_trimmed_audio_with_conversational_vad_params",
    "resampled_and_cut_audio_with_prompt_vad_params",
    "resampled_and_trimmed_audio_with_prompt_vad_params",
]

DATASET_VERSION_METADATA = {
    "v1": {"n_folds": 5, "uri": None},
    "v1.0.1": {"n_folds": 5, "uri": None},
    "v2.0": {"n_folds": 5, "uri": None},
    "v2.0.1": {"n_folds": 5, "uri": None},
    "v2.1": {"n_folds": 5, "uri": None},
    "v2.1.1": {"n_folds": 5, "uri": None},
    "v2.2": {"n_folds": 5, "uri": None},
    "v2.2.1": {"n_folds": 5, "uri": None},
    "v2.3.1": {"n_folds": 5, "uri": None},
    "v2.4.1": {"n_folds": 5, "uri": None},
    "v2.5.1": {"n_folds": 5, "uri": None},
    "v2.6.1": {"n_folds": 5, "uri": None},
    "v2.7.1": {"n_folds": 5, "uri": None},
    "v2.8.1": {"n_folds": 5, "uri": None},
    "v2.9.1": {"n_folds": 5, "uri": None},
    "v2.9.2": {"n_folds": 5, "uri": None},
    "v2.9.3": {"n_folds": 5, "uri": None},
    "v2.9.4": {"n_folds": 5, "uri": None},
    "v2.9.5": {"n_folds": 5, "uri": None},
    "v2.10.1": {"n_folds": 5, "uri": None},
    "v2.10.4": {"n_folds": 5, "uri": None},
    "v2.11.1": {"n_folds": 5, "uri": None},
    "v2.11.4": {"n_folds": 5, "uri": None},
}
# HARDCODED: Kintsugi remote storage URIs, blanked to None in this copy.
FEATURE_STORE_URI = None
RAW_AUDIO_URI = None


# Defines where audio and metadata is cached
# HARDCODED: local cache under ~/.kintsugi and ~/.cache/kintsugi.
CACHE_DIR = Path.home() / ".kintsugi"
AUDIO_DIR = CACHE_DIR / "audio"
METADATA_DIR = CACHE_DIR / "metadata"
USER_CACHE_DIR_KINTSUGI = Path.home() / ".cache" / "kintsugi"


class DatasetFields:
    """Key names of a batch dict: uuid, features, label.<task>.primary/kd, length, loss."""

    UUID = "uuid"
    FEATURES = "features"
    LABEL = "label"
    LENGTH = "length"
    LOSS = "loss"

    class LabelFields:
        """Sub-keys under label.<task>: primary (label column) and kd (scores_<task> column)."""

        PRIMARY = "primary"
        KD = "kd"


# HARDCODED: fixed 80-bin log-mel mean profile; WhisperTorchFeatureExtractor shifts each
# window's per-bin time mean to these values (channel equalization).
IDEAL_LOGMEL_ENERGIES = np.array(
    [
        0.34912264,
        0.58558977,
        0.7912451,
        0.92767584,
        0.98273695,
        0.98439455,
        0.9603633,
        0.93906444,
        0.9366281,
        0.93200225,
        0.916437,
        0.8928787,
        0.8637211,
        0.83265126,
        0.79977655,
        0.7778334,
        0.7561299,
        0.72997606,
        0.70391226,
        0.6800474,
        0.65755,
        0.63536274,
        0.61355984,
        0.5923383,
        0.5720056,
        0.55244887,
        0.53684795,
        0.5221597,
        0.5098636,
        0.49923953,
        0.48908615,
        0.47840047,
        0.46758702,
        0.47343993,
        0.46268672,
        0.4475126,
        0.46747103,
        0.45131385,
        0.4635319,
        0.44889897,
        0.45491976,
        0.4373785,
        0.43154317,
        0.42194438,
        0.41158468,
        0.40096927,
        0.3933149,
        0.38795966,
        0.38441542,
        0.38454026,
        0.3815766,
        0.3768835,
        0.3719921,
        0.3654539,
        0.35399568,
        0.3425986,
        0.32823247,
        0.31404305,
        0.30564603,
        0.29617435,
        0.29273877,
        0.28560263,
        0.27459458,
        0.26876706,
        0.25825337,
        0.24759005,
        0.24090728,
        0.2344712,
        0.22529823,
        0.20880115,
        0.193578,
        0.18290243,
        0.17621627,
        0.17087021,
        0.16641389,
        0.15932252,
        0.14312662,
        0.11790597,
        0.08030523,
        0.03747071,
    ],
    dtype=np.float32,
)
