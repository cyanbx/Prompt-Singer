# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import io
import csv
import logging
import re
import pickle
from torch import nn
from collections import defaultdict
from pathlib import Path
import random
from random import sample
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from fairseq.data import ConcatDataset, Dictionary, FairseqDataset, ResamplingDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.audio_utils import (
    parse_path, read_from_stored_zip, io,
)
from fairseq.data.audio.data_cfg import get_config_from_yaml
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
from einops import rearrange
from transformers import T5Tokenizer, T5EncoderModel

from research.PromptSinger.dataset.tokenizer.soundstream.AudioTokenizer import AudioTokenizer

logger = logging.getLogger(__name__)


def get_features_or_waveform(path: str):
    _path, slice_ptr = parse_path(path)
    assert _path.endswith(".zip")
    data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
    f = io.BytesIO(data)
    features_or_waveform = np.load(f, allow_pickle=True)
    return features_or_waveform


class T2ASingT5DataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path: Path):
        self.config = get_config_from_yaml(yaml_path)
        self.root = os.path.dirname(yaml_path)

    def _auto_convert_to_abs_path(self, x):
        if isinstance(x, str):
            if Path(x).exists():
                return x
            if Path(self.root + "/" + x).exists():
                return self.root + "/" + x
        elif isinstance(x, dict):
            return {k: self._auto_convert_to_abs_path(v) for k, v in x.items()}
        return x

    @property
    def uni_dict(self):
        return self.config.get("uni_dict", "dict_uni.txt")

    @property
    def word_dict(self):
        return self.config.get("word_dict", "gender_word_dict.pkl")

    @property
    def sentence_dict(self):
        return self.config.get("sentence_dict", "gender_sen_dict.pkl")
    
    @property
    def audio_tokenizer_ckpt_path(self):
        """soundstream checkpoint path"""
        return self.config.get("audio_tokenizer_ckpt_path", None)
    
    @property
    def text_encoder_version(self):
        """T5 repository path"""
        return self.config.get("text_encoder_version", None)
    

    @property
    def dict_length(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("dict_length", 5250)


    @property
    def shuffle(self) -> bool:
        """Shuffle dataset samples before batching"""
        return self.config.get("shuffle", False)

    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("pre_tokenizer", {"tokenizer": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("bpe_tokenizer", {"bpe": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_tgt_lang_tag", False)

    @property
    def prepend_bos_and_append_tgt_lang_tag(self) -> bool:
        """Prepend BOS and append target lang ID token to the target (e.g. mBART with language token pretraining)."""
        return self.config.get("prepend_bos_and_append_tgt_lang_tag", False)

    @property
    def input_feat_per_channel(self):
        """The dimension of input features (per audio channel)"""
        return self.config.get("input_feat_per_channel", 80)

    @property
    def input_channels(self):
        """The number of channels in the input audio"""
        return self.config.get("input_channels", 1)

    @property
    def sample_rate(self):
        return self.config.get("sample_rate", 16_000)

    @property
    def sampling_alpha(self):
        """Hyper-parameter alpha = 1/T for temperature-based resampling.
        (alpha = 1 for no resampling)"""
        return self.config.get("sampling_alpha", 1.0)

    @property
    def use_audio_input(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("use_audio_input", False)
    @property
    def apply_ucmvn(self) -> bool:
        return self.config.get("apply_ucmvn", True)

    @property
    def use_sample_rate(self):
        """Needed by the dataset loader to see if the model requires
        raw audio with specific sample rate as inputs."""
        return self.config.get("use_sample_rate", 16000)

    @property
    def batch_max_frames(self):
        return self.config.get("batch_max_frames", 800)

    @property
    def text_enc_dim(self):
        return self.config.get("text_enc_dim", 1024)
    
    @property
    def prompt_latent_length(self):
        return self.config.get("prompt_latent_length", 77)

    @property
    def max_dur(self):
        return self.config.get("max_dur", 200)

    @property
    def num_coarse_quantizers(self):
        return self.config.get("num_coarse_quantizers", 3)

    @property
    def audio_root(self):
        """Audio paths in the manifest TSV can be relative and this provides
        the root path. Set this to empty string when using absolute paths."""
        return self.config.get("audio_root", "")

    @property
    def global_cmvn_stats_npz(self) -> Optional[str]:
        path = self.config.get("global_cmvn", {}).get("stats_npz_path", None)
        return self._auto_convert_to_abs_path(path)

    @property
    def hub(self) -> Dict[str, str]:
        return self.config.get("hub", {})

    @property
    def external_nllb_code(self) -> Optional[str]:
        """Whether to map lang token to iso code in manifests"""
        return self.config.get("external_nllb_code", None)
    
    
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    

class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length  
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state.cpu().detach()
        
        return z

    @torch.no_grad()
    def encode(self, text):
        return self(text)
    

class T2ASingT5Dataset(FairseqDataset):
    TAG_TEMPLATE = "<>"

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: T2ASingT5DataConfig,
        samples: List[Dict],
        uni_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        batch_max_frames=400,
        num_coarse_quantizers=3,
        append_eos=False,
        is_generate=False
    ):
        print("num_coarse_quantizers:", num_coarse_quantizers)
        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.dict = uni_dict
        
        assert len(self.dict) == self.cfg.dict_length + 4, f"{len(self.dict)}, {self.cfg.dict_length}"
        
        self.samples = samples
        
        self.batch_max_frames = batch_max_frames
        self.num_coarse_quantizers = num_coarse_quantizers
        
        self.prompt_latent_length = self.cfg.prompt_latent_length
        
        self.n_frames = []
        logger.info("calculating data items lengths...")

        for i, item in enumerate(self.samples):
            self.n_frames.append(self.cal_n_frames(item))

        self.n_samples = len(self.samples)
        
        self.uni_dict = uni_dict
        self.spec_token_types = [
                                 'text_to_acoustic_sing',
                                 'continuous_token',
                                 'text_start', 'text_end',
                                 'prompt_start', 'prompt_end',
                                 'acoustic_start', 'acoustic_end',
                                 'f0_rescale_start', 'f0_rescale_end',
                                 'f0_bias_start', 'f0_bias_end'
                                ]
        
        self.spec_tokens = {token_type: torch.LongTensor([self.get_tag_idx("<{}>".format(token_type), self.dict)]) for token_type in self.spec_token_types}
        self.shuffle = cfg.shuffle if is_train_split else False

        self.pre_tokenizer = pre_tokenizer # None
        self.bpe_tokenizer = bpe_tokenizer # None
        self.n_frames_per_step = 1
        
        self.append_eos = append_eos
            
        def load_variable(fpath):
            f = open(fpath,'rb')
            r = pickle.load(f)
            f.close()
            return r
        
        self.is_generate = is_generate

        try:
            self.stage1_dict = load_variable(os.path.join(self.cfg.root, self.cfg.word_dict))
            self.stage2_dict = load_variable(os.path.join(self.cfg.root, self.cfg.sentence_dict))
        except FileNotFoundError:
            print('WARNING: Failed to load files of keywords and sentence templates. Only supports executing inference.')
            assert self.is_generate

        self.volume_type = {
            'low': [0.02, 0.04],
            'medium': [0.07, 0.10],
            'high': [0.16, 0.20]
        }
        
        self.text_encoder = FrozenT5Embedder(version=self.cfg.text_encoder_version)
        self.audio_tokenizer = AudioTokenizer(ckpt_path=self.cfg.audio_tokenizer_ckpt_path, device=torch.device('cuda'))
        
        logger.info(self.__repr__())

    
    def cal_n_frames(self, item):
        n_frame = self.batch_max_frames if int(item['n_frames']) > self.batch_max_frames else int(item['n_frames']) 
        n_frame_total = (n_frame * 3 + self.prompt_latent_length) * self.num_coarse_quantizers + 13 * self.num_coarse_quantizers # f0, semantic, acoustic, ref, others

        return n_frame_total


    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"shuffle={self.shuffle}, "
            f"n_frames_per_step={self.n_frames_per_step}"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    @classmethod
    def get_tag_idx(
        cls, tag: str, dictionary: Dictionary):
        tag_idx = dictionary.index(tag)
        assert tag_idx != dictionary.unk(), tag
        return tag_idx

    @classmethod
    def tokenize(cls, tokenizer, text: str):
        return text if tokenizer is None else tokenizer.encode(text)
    
    
    def get_duration_from_text(self, dur_str:str):
        dur_list = ['dur_' + item.strip() for item in dur_str[1:-1].split(',')]
        out_dur_str = ' '.join(dur_list)
        return out_dur_str
    

    def choose(self, gender=None, volume=None, pitch=None, p1=0, p2=0):
        
        def return_numbers_3(p1,p2):  
            if (p1+p2>1):
                print('error prob')
            random_num = random.random() 
            if random_num < p2:  
                return random.choice([['gender','volume'],['gender','pitch']])
            elif random_num < p1 + p2:  
                return random.choice(['gender','volume'])
            else:
                return None

        def return_numbers_2(a,b,p1):   
            random_num = random.random() 
            if random_num < p1:  
                return random.choice([a, b])
            else:
                return None  

        word_dict = self.stage1_dict
        sen_dict = self.stage2_dict
        
        sentence_list = []
        sentence = ''
        gender_word = random.choice(word_dict[f'{gender}_gender']) if gender is not None else ''  
        volume_word = random.choice(word_dict[f'{volume}_volume']) if volume is not None and volume != 'None' else ''  
        pitch_word = random.choice(word_dict[f'{pitch}_pitch']) if pitch is not None else '' 
        
        non_none_count = sum(1 for attribute in (gender, volume, pitch) if attribute is not None and attribute != 'None')
        
        if non_none_count == 0:
            return ''
        
        #1 word given
        elif non_none_count == 1:
            if gender!=None:
                sentence_list = sen_dict['gender']
            elif volume!=None and volume != 'None':
                sentence_list = sen_dict['volume_g'] + sen_dict[f'volume_{volume}']
        
        #2 words given
        elif non_none_count == 2:
            if volume == None or volume == 'None':
                if p1!=0:
                    random_num = random.random() 
                    if random_num < p1:  
                        sentence_list = sen_dict['gender']
                if sentence_list == []:
                    sentence_list = sen_dict['gender_pitch']
        
            elif pitch == None:
                if p1!=0:
                    choose_list = return_numbers_2('volume','gender',p1)
                    if choose_list!=None:
                        if 'gender' in choose_list:
                            sentence_list = sen_dict['gender']
                        elif 'volume' in choose_list:
                            sentence_list = sen_dict['volume_g'] + sen_dict[f'volume_{volume}']

                if sentence_list == []:
                    sentence_list = sen_dict['gender_volume_g'] + sen_dict[f'gender_volume_{volume}']
                                
        #3 words given    
        else:
            if p1!=0 or p2!=0:
                choose_list = return_numbers_3(p1,p2)
                if choose_list!=None:
                    if 'gender' in choose_list:  
                        if 'pitch' in choose_list:  
                            sentence_list = sen_dict['gender_pitch']
                        elif 'volume' in choose_list:  
                            sentence_list = sen_dict['gender_volume_g'] + sen_dict[f'gender_volume_{volume}']
                        else:  
                            sentence_list = sen_dict['gender']
                    elif 'volume' in choose_list:  
                        sentence_list = sen_dict['volume_g'] + sen_dict[f'volume_{volume}']

            if sentence_list == []:
                sentence_list = sen_dict['gender_volume_pitch_g'] + sen_dict[f'gender_volume_pitch_{volume}']
        
        sentence = random.choice(sentence_list)
        
        sentence = sentence.replace('[gender]', gender_word)    
        sentence = sentence.replace('[volume]', volume_word)
        sentence = sentence.replace('[pitch]', pitch_word)

        return sentence


    def __getitem__(self, index: int):
        # original sample content: item_name, task, lang, phone, dur, txt, semantic, n_frames
        sample = self.samples[index]
        
        sample_encoded = {'index': index }
        sample_encoded['item_name'] = sample['item_name']
        sample_encoded['task'] = torch.LongTensor([self.get_tag_idx(f"<{sample['task']}>", self.uni_dict)])
        sample_encoded['n_frames'] = sample['n_frames']
        
        if sample['task'] == 'text_to_acoustic_sing':
            phone = sample['phone']
            phone = " ".join([ph for ph in phone.split()])
        
            phone_encoded = self.uni_dict.encode_line(phone, add_if_not_exist=False, append_eos=False).long()
            sample_encoded['phone'] = phone_encoded

            def audio_rms(wav: torch.Tensor):
                return torch.sqrt(torch.mean(torch.square(wav), axis=-1)).item()
            
            if not self.is_generate:
                wav, sr = torchaudio.load(sample['audio_path'])
                if sr != self.audio_tokenizer.sr:
                    wav = torchaudio.transforms.Resample(sr, self.audio_tokenizer.sr)(wav)
            else:
                wav = torch.zeros((1, phone_encoded.shape[-1] * 320), dtype=torch.float)
            
            if 'volume' in sample:
                volume_label = sample['volume']
            else:
                volume_label = random.choice(['low', 'medium', 'high'])
            
            gender_label = sample['gender']
            pitch_label = sample['pitch']

            if 'prompt' in sample:
                overall_text_prompt = sample['prompt']
            else:
                assert not self.is_generate
                if 'volume' not in sample:
                    overall_text_prompt = self.choose(gender = gender_label, volume = volume_label, pitch = pitch_label, p1 = 0.05, p2 = 0.05)
                else:
                    overall_text_prompt = self.choose(gender = gender_label, volume = volume_label, pitch = pitch_label, p1 = 0.0, p2 = 0.0)
            
            sample_encoded['prompt'] = self.text_encoder.encode(overall_text_prompt).squeeze(0)

            original_rms = audio_rms(wav)
            
            if volume_label != 'None':
                rms_rescaled = random.uniform(self.volume_type[volume_label][0], self.volume_type[volume_label][1])
                wav_rescaled = wav / original_rms * rms_rescaled
                wav_rescaled = torch.clamp(wav_rescaled, -0.99, 0.99)
            else:
                wav_rescaled = wav
            
            acoustic_data = self.audio_tokenizer.encode(wav_rescaled)
            
            acoustic_encoded_lines = []
            for code_idx, tokenized in enumerate(acoustic_data[:self.num_coarse_quantizers]):
                _tokenized = " ".join(["aco_" + str(token + code_idx * 1024) for token in tokenized])
                target_ = self.uni_dict.encode_line(_tokenized, add_if_not_exist=False, append_eos=False).long()
                acoustic_encoded_lines.append(target_)
            
            acoustic_encoded = torch.stack(acoustic_encoded_lines).transpose(0, 1)
            sample_encoded['acoustic'] = acoustic_encoded

            f0_rescale = np.array([int(x) for x in sample['f0_rescale'].split()])
            f0_rescale[f0_rescale > 850] = 850
            f0_rescale[f0_rescale < 0] = 0
            f0_rescale_str = " ".join(["f0_" + str(idx) for idx in f0_rescale])
            f0_rescale_encoded = self.uni_dict.encode_line(f0_rescale_str, add_if_not_exist=False, append_eos=False).long()
            sample_encoded['f0_rescale'] = f0_rescale_encoded

            f0_bias_str = "f0_" + sample['f0_avg']
            f0_bias_encoded = self.uni_dict.encode_line(f0_bias_str, add_if_not_exist=False, append_eos=False).long()
            sample_encoded['f0_bias'] = f0_bias_encoded

            min_len = min(sample_encoded['phone'].shape[0], sample_encoded['f0_rescale'].shape[0])
            min_len = min(min_len, sample_encoded['acoustic'].shape[0])

            sample_encoded['f0_rescale'] = sample_encoded['f0_rescale'][:min_len]
            sample_encoded['phone'] = sample_encoded['phone'][:min_len]
            sample_encoded['acoustic'] = sample_encoded['acoustic'][:min_len]

        else:
            raise NotImplementedError(f"task {sample['task']} not implemented")
        
        v_detach = lambda x: x.detach() if isinstance(x, torch.Tensor) else x

        sample_encoded = {
            k: v_detach(v) for k,v in sample_encoded.items()
        }

        return sample_encoded
    

    def __len__(self):
        return self.n_samples

    def crop_to_max_size(self, token, target_size, start=None):
        size = len(token)
        diff = size - target_size
        if diff <= 0:
            return token, 0
        # longer utterances
        if start is None:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        else:
            end = start + target_size
        return token[start:end], start


    def process_units(self, units, reduce=False):
        return units
    

    def collator_token(self, samples):
        collated_sources, collated_targets = [], []
        target_loss_masks, source_prefix_masks = [], []
        item_sizes = []
        ref_sequences = []

        for i, sample in enumerate(samples):
            if sample['task'] == self.spec_tokens['text_to_acoustic_sing']:
                phone, acoustic, f0_rescale = sample['phone'],  sample['acoustic'], sample['f0_rescale']
                if len(phone) > self.batch_max_frames:
                    phone, start = self.crop_to_max_size(phone, self.batch_max_frames, None if self.is_train_split else 0)
                    acoustic, _ = self.crop_to_max_size(acoustic, round(self.batch_max_frames), round(start))
                    f0_rescale, _ = self.crop_to_max_size(f0_rescale, self.batch_max_frames, start)

                acoustic = rearrange(acoustic, 'l q -> (l q)').long()

                phone = torch.flatten(torch.stack([phone] * self.num_coarse_quantizers, dim=1))
                f0_rescale = torch.flatten(torch.stack([f0_rescale] * self.num_coarse_quantizers, dim=1))

                f0_bias = sample['f0_bias']
                f0_bias = torch.flatten(torch.stack([f0_bias] * self.num_coarse_quantizers, dim=1))
                
                target_sequence = torch.cat(
                    [self.spec_tokens['text_to_acoustic_sing']] * self.num_coarse_quantizers + 
                    [self.spec_tokens['prompt_start']] * self.num_coarse_quantizers + 
                    [self.spec_tokens['continuous_token']] * (self.prompt_latent_length * self.num_coarse_quantizers) + 
                    [self.spec_tokens['prompt_end']] * self.num_coarse_quantizers +
                    [self.spec_tokens['f0_rescale_start']] * self.num_coarse_quantizers +
                    [f0_rescale] +
                    [self.spec_tokens['f0_rescale_end']] * self.num_coarse_quantizers +
                    [self.spec_tokens['text_start']] * self.num_coarse_quantizers +
                    [phone] + 
                    [self.spec_tokens['text_end']] * self.num_coarse_quantizers + 
                    [self.spec_tokens['f0_bias_start']] * self.num_coarse_quantizers +
                    [f0_bias] +
                    [self.spec_tokens['f0_bias_end']] * self.num_coarse_quantizers + 
                    [self.spec_tokens['acoustic_start']] * self.num_coarse_quantizers +
                    [acoustic] + 
                    [self.spec_tokens['acoustic_end']] * self.num_coarse_quantizers + 
                    [torch.LongTensor([self.dict.eos()])] * self.num_coarse_quantizers)
  
                source_prefix_mask = torch.zeros_like(target_sequence, dtype=torch.bool)
                source_prefix_mask[:self.prompt_latent_length * self.num_coarse_quantizers +f0_rescale.shape[0]+phone.shape[0] + self.num_coarse_quantizers * 8] = True
                target_loss_mask = torch.zeros_like(target_sequence, dtype=torch.bool)
                target_loss_mask[self.prompt_latent_length * self.num_coarse_quantizers+f0_rescale.shape[0]+phone.shape[0] + self.num_coarse_quantizers * 7: ] = True
            
            else:
                raise NotImplementedError("task type not implemented")

            source_sequence = target_sequence

            collated_targets.append(target_sequence)
            collated_sources.append(source_sequence)
            source_prefix_masks.append(source_prefix_mask)
            target_loss_masks.append(target_loss_mask)
            item_sizes.append(target_sequence.shape[0])
            ref_sequences.append(sample['prompt'])

        return collated_sources, collated_targets, source_prefix_masks, target_loss_masks, item_sizes, ref_sequences


    def collater(
        self, samples: List[Dict], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        
        indices = torch.tensor([x['index'] for x in samples], dtype=torch.long)
        item_names = [x['item_name'] for x in samples]
        
        sources, targets, source_prefix_masks, target_loss_masks, total_sizes, ref_sequences = self.collator_token(samples)
            
        src_tokens = fairseq_data_utils.collate_tokens(
            sources,
            self.uni_dict.pad(),
            self.uni_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )

        target = fairseq_data_utils.collate_tokens(
            targets,
            self.uni_dict.pad(),
            self.uni_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )

        ref_seq = _collate_frames(ref_sequences)

        target_loss_mask = fairseq_data_utils.collate_tokens(
            target_loss_masks,
            False,
            None,
            left_pad=False,
            move_eos_to_beginning=False
        )

        source_prefix_mask = fairseq_data_utils.collate_tokens(
            source_prefix_masks,
            False,
            None,
            left_pad=False,
            move_eos_to_beginning=False
        )

        total_size_tensor = torch.tensor(total_sizes, dtype=torch.long)
        src_lengths, order = total_size_tensor.sort(descending=True)
        src_tokens = src_tokens.index_select(0, order)
        target = target.index_select(0, order)
        ref_seq = ref_seq.index_select(0, order)
        indices = indices.index_select(0, order)

        if order.shape[0] == 1:
            item_names = [item_names[order]]
            
        else:
            item_names = [item_names[o] for o in order]
            
        target_loss_mask = target_loss_mask.index_select(0, order)
        source_prefix_mask = source_prefix_mask.index_select(0, order)
        
        net_input = {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "target_acoustic_mask": target_loss_mask,
            "text_feature": ref_seq
        }
        
        return_dict = {
            "id": indices,
            "item_names": item_names,
            "net_input": net_input,
            "target": target,
            "nsentences": len(samples),
            "ntokens": sum(total_sizes),
            "source_prefix_mask": source_prefix_mask,
            "target_acoustic_mask": target_loss_mask
        }

        return return_dict


    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        return self.n_frames[index] #, self.tgt_lens[index]

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])

        return np.lexsort(order)

    def prefetch(self, indices):
        raise False


class T2ASingT5DatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SEMANTIC, KEY_N_FRAMES = "item_name", "semantic", "n_frames"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

    @classmethod
    def get_size_ratios(
        cls, datasets: List[T2ASingT5Dataset], alpha: float = 1.0
    ) -> List[float]:
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""

        id_to_lp, lp_to_sz = {}, defaultdict(int)
        for ds in datasets:
            lang_pairs = {f"{s}->{t}" for s, t in zip(ds.src_langs, ds.tgt_langs)}
            assert len(lang_pairs) == 1
            lang_pair = list(lang_pairs)[0]
            id_to_lp[ds.split] = lang_pair
            lp_to_sz[lang_pair] += sum(ds.n_frames)

        sz_sum = sum(v for v in lp_to_sz.values())
        lp_to_prob = {k: v / sz_sum for k, v in lp_to_sz.items()}
        lp_to_tgt_prob = {k: v**alpha for k, v in lp_to_prob.items()}
        prob_sum = sum(v for v in lp_to_tgt_prob.values())
        lp_to_tgt_prob = {k: v / prob_sum for k, v in lp_to_tgt_prob.items()}
        lp_to_sz_ratio = {
            k: (lp_to_tgt_prob[k] * sz_sum) / v for k, v in lp_to_sz.items()
        }
        size_ratio = [lp_to_sz_ratio[id_to_lp[ds.split]] for ds in datasets]

        p_formatted = {
            k: f"{lp_to_prob[k]:.3f}->{lp_to_tgt_prob[k]:.3f}" for k in lp_to_sz
        }
        logger.info(f"sampling probability balancing: {p_formatted}")
        sr_formatted = {ds.split: f"{r:.3f}" for ds, r in zip(datasets, size_ratio)}
        logger.info(f"balanced sampling size ratio: {sr_formatted}")
        return size_ratio
    
    @classmethod
    def _load_samples_from_tsv(cls, root: str, split: str, drop_probability=0.0):
        tsv_path = Path(root) / f"{split}.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            samples = [dict(e) for e in reader]
            if drop_probability > 1e-6:
                samples = sample(samples, int(len(samples) * (1 - drop_probability)))
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {tsv_path}")
        return samples

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: T2ASingT5DataConfig,
        split: str,
        uni_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        batch_max_frames,
        drop_probability: float = 0.0,
        is_generate=False
    ) -> T2ASingT5Dataset:
        samples = cls._load_samples_from_tsv(
            root, split, drop_probability=drop_probability
        )

        dataset_cls = T2ASingT5Dataset
        num_coarse_quantizers = int(cfg.num_coarse_quantizers)

        return dataset_cls(
            split=split,
            is_train_split=is_train_split,
            cfg=cfg,
            samples=samples,
            uni_dict=uni_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            batch_max_frames=batch_max_frames,
            num_coarse_quantizers=num_coarse_quantizers,
            is_generate=is_generate
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: T2ASingT5DataConfig,
        splits: str,
        uni_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        batch_max_frames: int,
        drop_probability=0.0,
        is_generate=False
    ) -> T2ASingT5Dataset:
        datasets = [
            cls._from_tsv(
                root,
                cfg,
                split,
                uni_dict,
                is_train_split,
                pre_tokenizer,
                bpe_tokenizer,
                batch_max_frames,
                drop_probability=(drop_probability if "train" in split else 0.0),
                is_generate=is_generate
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

