# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Feature extractor class for Whisper in native torch code. Modified from
`transformers.models.whisper.feature_extraction_whisper`.
"""
import torch
from transformers.audio_utils import mel_filter_bank

from kirad.constants import IDEAL_LOGMEL_ENERGIES


class WhisperTorchFeatureExtractor(torch.nn.Module):
    r"""
    Constructs a Whisper feature extractor.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the Fourier transform.
        dither (`float`, *optional*, defaults to 0.0):
            Adds dithering. In other words, adds a small Gaussian noise to each frame.
            E.g. use 0.0001 to add dithering with a normal distribution centered
            around 0.0 with standard deviation 0.0001 (assuming [-1,+1] range of raw_speech).
            The value 0.0 means no dithering.
            Dithering has similar effect as `spectrogram(mel_floor=...)`. It reduces
            the high log_mel_fbank values for signals with hard-zero sections,
            when VAD cutoff is present in the signal.
        ideal_logmel_energies (bool, *optional*, defaults to True):
            Whether to use ideal_logmel_energies normalization
        device (`str`, *optional*, defaults to `'cpu'`):
            Specifies the device for computation of the log-mel spectrogram of audio signals in the
            `_torch_extract_fbank_features` method. (e.g., "cpu", "cuda")
    """

    def __init__(
        self,
        feature_size=80,
        hop_length=160,
        n_fft=400,
        dither=0.0,
        normalize_audio=True,
        ideal_logmel_energies=True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dither = dither
        self.normalize_audio = normalize_audio
        self.ideal_logmel_energies = ideal_logmel_energies
        self.window = torch.hann_window(self.n_fft)
        self.mel_filters = torch.from_numpy(
            mel_filter_bank(
                num_frequency_bins=1 + n_fft // 2,
                num_mel_filters=feature_size,
                min_frequency=0.0,
                max_frequency=8000.0,
                sampling_rate=16000,
                norm="slaney",
                mel_scale="slaney",
            )
        )
        # To enable model compilation, ideal_logmel_energies should be registered as a buffer or Parameter
        self.register_buffer(
            "ideal_logmel_energies_stats",
            torch.from_numpy(IDEAL_LOGMEL_ENERGIES).float(),
        )

    def _torch_extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-mel spectrogram of the audio using PyTorch's GPU-accelerated STFT implementation with batching,
        yielding results similar to cpu computing with 1e-5 tolerance.
        """

        # Note: it would be better to dither the chunked waveform,
        # so overlapping signal does not get the same dithering.
        # But, chunking is happening inside pytorch, so it is here.
        if self.dither != 0.0:
            waveform += self.dither * torch.randn(
                waveform.shape, dtype=waveform.dtype, device=waveform.device
            )

        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=self.window.to(waveform),
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = self.mel_filters.to(magnitudes).T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0

    def forward(
        self,
        batched_speech: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mel filter bank features from a batch x samples tensor of audio."""
        # Got NaNs when doing feature computations in half or single precision
        batched_speech_double = batched_speech.double()

        # zero-mean and unit-variance normalization
        if self.normalize_audio:
            batched_speech_double = (
                batched_speech_double - batched_speech_double.mean(dim=1, keepdim=True)
            ) / torch.sqrt(batched_speech_double.var(dim=1, keepdim=True) + 1e-7)
        features = self._torch_extract_fbank_features(batched_speech_double)

        if self.ideal_logmel_energies:
            features += self.ideal_logmel_energies_stats.to(features)[
                None, :, None
            ] - features.mean(dim=-1, keepdim=True)

        return features.to(batched_speech)
