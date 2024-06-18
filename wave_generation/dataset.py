# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.
import warnings
warnings.filterwarnings("ignore")
import math
import os
import csv
import random
import torch
import torch.utils.data
import numpy as np
import pickle
from copy import deepcopy
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm


MAX_WAV_VALUE = 32768.0


def load_wav(full_path, sr_target):
    sampling_rate, data = read(full_path)
    if sampling_rate != sr_target:
        raise RuntimeError("Sampling rate of the file {} is {} Hz, but the model requires {} Hz".
              format(full_path, sampling_rate, sr_target))
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class UnitDataset(torch.utils.data.Dataset):
    def __init__(self, data_manifest, acoustic_data_path, hparams, codebook_num, segment_size, unit_hop_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, is_seen=True):
        
        super().__init__()
        
        data_manifest = open(data_manifest)
        data_reader = csv.DictReader(
            data_manifest,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )

        self.data_items = [e for e in data_reader]
        self.acoustic_file = np.load(acoustic_data_path, allow_pickle=True).item()
        self.name = 'Prompt_sing'

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.unit_hop_size = unit_hop_size
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.codebook_num = codebook_num


    def __getitem__(self, index):
        '''
        RETURN:
        unit: [codebook_num, Tu]
        audio: pre-extracted, [Ta]  
        filename: pre_extracted, 
        mel_for_loss: calculated online
        '''
        #ipdb.set_trace()
        data_item = self.data_items[index]
        filename = data_item['item_name'] + '.wav'
        audio, _ = load_wav(data_item['path'], self.sampling_rate)
        #audio = audio / MAX_WAV_VALUE
        audio = audio / 1.0

        audio = torch.FloatTensor(audio).unsqueeze(0) # [1, Ta]
        acoustic_tokens = torch.from_numpy(self.acoustic_file[data_item['item_name']]) 
        assert acoustic_tokens.shape[0] >= self.codebook_num, f'codebook_num {self.codebook_num} exceeds acoustic token codebook num {acoustic_tokens.shape[0]}'
        acoustic_tokens = acoustic_tokens[:self.codebook_num, ...]# [codebook_num, Tu]

        # do size matching first
        code_length = min(audio.shape[-1] // self.unit_hop_size, acoustic_tokens.shape[-1])
        acoustic_tokens = acoustic_tokens[:, :code_length]
        audio = audio[:, :code_length * self.unit_hop_size]

        if self.split:
            while audio.shape[-1] < self.segment_size:
                audio = torch.hstack([audio, audio])
                acoustic_tokens = torch.hstack([acoustic_tokens, acoustic_tokens])

            # if audio.shape[-1] > self.segment_size:
            unit_segment_size = self.segment_size // self.unit_hop_size
            assert self.segment_size == unit_segment_size * self.unit_hop_size, f"segment_size {self.segment_size} cannot be devided by unit_hop_size{self.unit_hop_size}"

            max_unit_start = acoustic_tokens.shape[-1] - unit_segment_size
            
            unit_start = random.randint(0, max_unit_start)
            audio_start = unit_start * self.unit_hop_size
            
            acoustic_tokens = acoustic_tokens[:, unit_start:unit_start + unit_segment_size]
            audio = audio[:, audio_start: audio_start + self.segment_size]
        else:
            assert audio.shape[-1] == acoustic_tokens.shape[-1] * self.unit_hop_size, "audio shape {} acoustic shape {}".format(audio.shape, acoustic_tokens.shape)

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (acoustic_tokens.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.data_items)
