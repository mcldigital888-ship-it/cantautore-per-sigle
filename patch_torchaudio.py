"""
Patch per torchaudio.save e torchaudio.load: usa soundfile invece di torchcodec.
Importa questo modulo prima di qualsiasi codice che usa torchaudio.
"""
import numpy as np
import soundfile as sf
import torch
import torchaudio


def _patched_save(filepath, src, sample_rate, **kwargs):
    """Salva audio usando soundfile invece di torchcodec."""
    if isinstance(src, torch.Tensor):
        data = src.cpu().numpy()
        if data.ndim == 2:
            data = data.T
    else:
        data = src
    sf.write(str(filepath), data, sample_rate)


def _patched_load(filepath, **kwargs):
    """Carica audio usando soundfile invece di torchcodec."""
    data, sample_rate = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    else:
        data = data.T  # (samples, channels) -> (channels, samples)
    return torch.from_numpy(data), sample_rate


# Applica le patch
torchaudio.save = _patched_save
torchaudio.load = _patched_load
