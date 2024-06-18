# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from omegaconf import II

from fairseq.data import (
    Dictionary,
)
from fairseq.data import encoders
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.data.indexed_dataset import get_available_dataset_impl

from pathlib import Path
from research.PromptSinger.dataset.t2a_sing_t5_config_dataset import T2ASingT5DataConfig, T2ASingT5DatasetCreator


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


@dataclass
class AcousticLanguageModelingConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    config_yaml: Optional[str] = field(
        default='config.yaml', metadata={"help": "path to data config file"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_source_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False,
        metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False,
        metadata={"help": "boolean to pad to fixed batch size"},
    )
    ds_drop_probability: Optional[float] = field(
        default=0.0,
        metadata={"help": "dataset drop probability"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


@register_task("t2a_sing_t5_config_task", dataclass=AcousticLanguageModelingConfig)
class T2ASingT5ConfigTask(LegacyFairseqTask):

    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    def __init__(self, args, dictionary, targets=None):
        super().__init__(args)
        self.dict = dictionary
        
        self.data_cfg = T2ASingT5DataConfig(Path(args.data) / args.config_yaml)
        if (
            self.data_cfg.prepend_tgt_lang_tag
            and self.data_cfg.prepend_bos_and_append_tgt_lang_tag
        ):
            raise ValueError(
                "Please set only one of the two options to avoid adding target token multiple times"
            )
        
        self.batch_max_frames = self.data_cfg.batch_max_frames

        if targets is None:
            targets = ["future"]
        self.targets = targets


    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # dictionary, output_dictionary = cls.setup_dictionary(args, **kwargs)
        data_cfg = T2ASingT5DataConfig(Path(args.data) / args.config_yaml)
        dict_path = Path(args.data) / data_cfg.uni_dict
        
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        
        uni_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"unified dictionary size ({data_cfg.uni_dict}): " f"{len(uni_dict):,}"
        )
        
        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        # upgrade old checkpoints
        if getattr(args, "exclude_self_target", False):
            args.self_target = False

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, uni_dict, targets=targets)
    

    def build_model(self, args, from_checkpoint=False):
        args.n_vocab = self.data_cfg.dict_length + 4
        args.text_enc_dim = self.data_cfg.text_enc_dim
        model = super().build_model(args, from_checkpoint)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    f"Unsupported language modeling target: {target} not in {model.supported_targets}"
                )
                
        assert model.prompt_latent_length == self.data_cfg.prompt_latent_length
        return model
    
    
    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))


    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))
    

    def load_dataset(self, split: str, epoch=1, is_generate=False, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        
        self.datasets[split] = T2ASingT5DatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.uni_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            batch_max_frames=self.batch_max_frames,
            drop_probability=self.args.ds_drop_probability,
            is_generate=is_generate
        )
    

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the language_modeling task is not supported"
                )

            prefix_tokens = sample["net_input"]["src_tokens"]
            prefix_tokens.masked_fill_(~sample["source_prefix_mask"], self.dict.pad_index)
            prefix_max_length = (sample['source_prefix_mask']).to(dtype=torch.int).sum(dim=-1)
            
            prefix_max_length = prefix_max_length.max().item()
            prefix_tokens = prefix_tokens[:, :prefix_max_length]
            
            if prefix_tokens[:, 0].eq(bos_token).all():
                prefix_tokens = prefix_tokens[:, 1:]

            src_len = sample["net_input"]["src_lengths"][0]

            max_len = min(
                int(src_len) + 200,
                models[0].n_ctx - 1
            )

            min_len = prefix_tokens.shape[1]

            codebook_start_idx = self.dict.index('aco_0')
            codebook_end_idx = self.dict.index('aco_3071') + 1

            f0_start_idx = self.dict.index('f0_0') if self.dict.index('f0_0') != self.dict.unk() else self.dict.index('f0_1')
            f0_end_idx = self.dict.index('AP')
            
            acoustic_start_token = self.dict.index('<acoustic_start>')
            acoustic_end_token = self.dict.index('<acoustic_end>')
            f0_end_token = self.dict.index('<f0_bias_end>')

            return models[0].generate_bias(prefix_tokens, sample['net_input']['text_feature'], (max_len - prefix_tokens.shape[1]) // 3, (min_len - prefix_tokens.shape[1]) // 3, f0_start_idx, f0_end_idx, codebook_start_idx, codebook_end_idx, f0_end_token, acoustic_start_token, acoustic_end_token)

    
    @property
    def dictionary(self):
        return self.dict
    
    @property
    def uni_dict(self):
        return self.dict
    
    @property
    def target_dictionary(self):
        return self.dict
    
    @property
    def source_dictionary(self):
        return self.dict
