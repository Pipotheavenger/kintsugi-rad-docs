import copy

import numpy as np
import torch
from scipy import signal


###############################################################
# ========== Original RawBoost Functions ==========
# https://github.com/TakHemlata/RawBoost-antispoofing
###############################################################
def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y


def normWav(x, always):
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def genNotchCoeffs(
    nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b
        )

    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x, b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2) : int(y.shape[0] - N / 2)]
    return y


# Linear and non-linear convolutive noise
def LnL_convolutive_noise(
    x,
    N_f,
    nBands,
    minF,
    maxF,
    minBW,
    maxBW,
    minCoeff,
    maxCoeff,
    minG,
    maxG,
    minBiasLinNonLin,
    maxBiasLinNonLin,
    fs,
):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG - minBiasLinNonLin
            maxG = maxG - maxBiasLinNonLin
        b = genNotchCoeffs(
            nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
        )
        y = y + filterFIR(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    y = normWav(y, 0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(
        ((2 * np.random.rand(p.shape[0])) - 1), ((2 * np.random.rand(p.shape[0])) - 1)
    )
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y, 0)
    return y


# Stationary signal independent noise
def SSI_additive_noise(
    x,
    SNRmin,
    SNRmax,
    nBands,
    minF,
    maxF,
    minBW,
    maxBW,
    minCoeff,
    maxCoeff,
    minG,
    maxG,
    fs,
):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(
        nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
    )
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = (
        noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * SNR)
    )
    x = x + noise
    return x


class RawBoost:
    """
    Plug-and-play RawBoost augmentation (official defaults)
    The following are plug-and-play algorithm presets from the official paper:
    https://arxiv.org/abs/2111.04433
    Preset #4 seems to be the top-performing one.

    See the paper for the meaning of all other parameters
    algo:
      0 - None
      1 - LnL
      2 - ISD
      3 - SSI
      4 - LnL → ISD → SSI
      5 - LnL → ISD
      6 - LnL → SSI
      7 - ISD → SSI
      8 - LnL + ISD (parallel summed)
    """

    def __init__(
        self,
        algo=4,
        fs=16000,
        nBands=5,
        minF=20,
        maxF=8000,
        minBW=100,
        maxBW=1000,
        minCoeff=10,
        maxCoeff=100,
        minG=0,
        maxG=0,
        minBiasLinNonLin=5,
        maxBiasLinNonLin=20,
        N_f=5,
        P=10,
        g_sd=2,
        SNRmin=10,
        SNRmax=40,
    ):

        if algo > 8:
            raise ValueError(f"Invalid algo: {algo}")

        self.algo = algo
        self.fs = fs

        self.nBands = nBands
        self.minF = minF
        self.maxF = maxF
        self.minBW = minBW
        self.maxBW = maxBW
        self.minCoeff = minCoeff
        self.maxCoeff = maxCoeff
        self.minG = minG
        self.maxG = maxG
        self.minBias = minBiasLinNonLin
        self.maxBias = maxBiasLinNonLin
        self.N_f = N_f

        self.P = P
        self.g_sd = g_sd

        self.SNRmin = SNRmin
        self.SNRmax = SNRmax

    def __call__(self, x: torch.tensor):

        orig_dtype, orig_device = x.dtype, x.device
        x = x.numpy()

        # convenience function aliases
        LnL = lambda sig: LnL_convolutive_noise(
            sig,
            self.N_f,
            self.nBands,
            self.minF,
            self.maxF,
            self.minBW,
            self.maxBW,
            self.minCoeff,
            self.maxCoeff,
            self.minG,
            self.maxG,
            self.minBias,
            self.maxBias,
            self.fs,
        )

        ISD = lambda sig: ISD_additive_noise(sig, self.P, self.g_sd)

        SSI = lambda sig: SSI_additive_noise(
            sig,
            self.SNRmin,
            self.SNRmax,
            self.nBands,
            self.minF,
            self.maxF,
            self.minBW,
            self.maxBW,
            self.minCoeff,
            self.maxCoeff,
            self.minG,
            self.maxG,
            self.fs,
        )

        # ----- Algorithms -----
        if self.algo == 1:  # LnL
            x = LnL(x)

        if self.algo == 2:  # ISD
            x = ISD(x)

        if self.algo == 3:  # SSI
            x = SSI(x)

        if self.algo == 4:  # LnL → ISD → SSI
            x = SSI(ISD(LnL(x)))

        if self.algo == 5:  # LnL → ISD
            x = ISD(LnL(x))

        if self.algo == 6:  # LnL → SSI
            x = SSI(LnL(x))

        if self.algo == 7:  # ISD → SSI
            x = SSI(ISD(x))

        if self.algo == 8:  # parallel LnL + ISD
            x = LnL(x) + ISD(x)

        return torch.tensor(x, dtype=orig_dtype, device=orig_device)
