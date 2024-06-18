# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from einops import rearrange

from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

from fairseq.utils import safe_getattr
from tqdm import tqdm

logger = logging.getLogger(__name__)

torch.set_printoptions(profile="full")

DEFAULT_MAX_TARGET_POSITIONS = 1024


def attention_mask(loss_mask, prefix_lm=True):
    """
    Generate the attention mask from the loss mask,
    where the loss mask is in the format [Batch, Length].
    Usually, the loss mask would look like:
      <False> ... <True> ... <False>, which represents the
    prefix, the target sequence and padding respectively.

    This function generates the mask for multi-head attention,
    which is in the shape of [Batch, Length, Length] and features:
    (1) the prefix entries can see all each other, if prefix_lm,
        otherwise causal;
    (2) the target entries are causal to each other and can see all
        prefix entries;
    (3) the padding entries can neither been seen nor see all other
        entries.
    """

    # basic preparation
    device = loss_mask.device
    batch_size, q_len = loss_mask.size()
    axis = torch.arange(q_len).to(device)
    # find the start and end time indices of loss duration
    start = axis.unsqueeze(0).masked_fill(~loss_mask, 1e8).min(dim=1).values
    end = axis.unsqueeze(0).masked_fill(~loss_mask, -1e8).max(dim=1).values
    # we strictly require that there is only one continuous True segment
    # for each example in the loss_mask:
    assert torch.all(end - start == loss_mask.int().sum(dim=-1) - 1)

    # (1) make it causal
    mask = (axis.unsqueeze(1) >= axis.unsqueeze(0)).repeat(batch_size, 1, 1)
    # (2) allow non-causaility in prefix part, if prefix_lm
    if prefix_lm:
        mask = torch.where(start.view(batch_size, 1, 1) > axis.view(1, 1, q_len),
                       True, mask)

    # (3) kill the padding
    mask = torch.where(end.view(batch_size, 1, 1) < axis.view(1, 1, q_len),
                       False, mask)

    return mask


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x, masks=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if masks is not None:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=masks, dropout_p=self.dropout if self.training else 0, is_causal=False)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            if masks is not None:
                raise NotImplementedError("Can only be causal if flash attention is not supported.")
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias, dropout, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x, masks=None):
        x = x + self.attn(self.ln_1(x), masks)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class GPT_global(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer, dropout=0.0, bias=0.0, prefix_lm=False):
        super().__init__()
        assert n_vocab is not None
        assert n_ctx is not None
        num_quantizer = 3 
        self.n_ctx = n_ctx
        self.n_vocab = n_vocab
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(n_vocab, n_state // num_quantizer),
            wpe = nn.Embedding(n_ctx, n_state),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_state, n_head, bias, dropout, n_ctx) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_state, bias=bias)
        ))
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
        # report number of parameters
        print("number of parameters GLOBAL: %.2fM" % (self.get_num_params()/1e6,))

        self.prefix_lm = prefix_lm

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, embed, masks=None):
        if self.prefix_lm:
            assert masks is not None
            masks = attention_mask(masks, prefix_lm=True).unsqueeze(1)
        else:
            masks = None

        x = self.transformer.drop(embed)
        for block in self.transformer.h:
            x = block(x, masks)
        x = self.transformer.ln_f(x)
        return x


class GPT_local(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer, dropout=0.0, bias=0.0):
        super().__init__()
        assert n_vocab is not None
        assert n_ctx is not None
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(n_vocab, n_state),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_state, n_head, bias, dropout, n_ctx) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_state, bias=bias),
        ))
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
        # report number of parameters
        print("number of parameters LOCAL: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, embed):
        x = self.transformer.drop(embed)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x

    def get_embeding(self, idx):
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        return tok_emb
    

@dataclass
class AudioboxMegabyteConfig(FairseqDataclass):

    n_vocab: int = field(
        default=4078, metadata={"help": "number of embedding entries of megabyte"}
    )

    n_ctx: int = field(
        default=10000, metadata={"help": "max sequence length expected"}
    )

    n_state: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )

    n_head: int = field(
        default=12, metadata={"help": "number of attention head"}
    )

    n_layer: int = field(
        default=16, metadata={"help": "number of layers of global transformer"}
    )

    prefix_lm: bool = field(
        default=True, metadata={"help": "whether to use prefix-lm"}
    )

    na_repeat: int = field(
        default=3, metadata={"help": "patch size. and the number that non-acoustic tokens need to repeat"}
    )

    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})

    prompt_latent_length: int = field(
        default=77, metadata={"help": "length of perceiver resampler output (i.e. length of encoded prompt sequence)"}
    )

    text_enc_dim: int = field(
        default=1024, metadata={"help": "dim of text prompt encoder hidden state."}
    )
    

@register_model("acoustic_lm", dataclass=AudioboxMegabyteConfig)
class AcousticLMT5Configured(BaseFairseqModel):
    
    def __init__(self, args):
        super().__init__()
        self.pad = 0
        self.patch_size = args.na_repeat # 
        self.prompt_latent_length = args.prompt_latent_length
        self.bias = 0.0
        self.drop_out = args.dropout
        self.n_ctx = args.n_ctx
        self.global_model = GPT_global(n_vocab=args.n_vocab, n_ctx=args.n_ctx, n_state=args.n_state, n_head=args.n_head, n_layer=args.n_layer, prefix_lm=args.prefix_lm, dropout=args.dropout) # build the global model
        self.local_model = GPT_local(n_vocab=args.n_vocab, n_ctx=self.patch_size+1, n_state=args.n_state, n_head=8, n_layer=6, dropout=args.dropout) # build the local model
        self.linear_map = nn.Linear(args.n_state//self.patch_size, args.n_state)
        self.lm_head = nn.Linear(args.n_state, args.n_vocab, bias=False) # 
        
        self.ln = nn.Linear(args.text_enc_dim, args.n_state)

        # needed as a fairseq model
        self.onnx_trace = False
        self.adaptive_softmax = None

    # --- needed as a fairseq model ---
    @property
    def supported_targets(self):
        return {"future"}
    
    def max_positions(self):
        return self.n_ctx
    
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
    
    @classmethod
    def build_model(cls, args, task):
        return cls(args)


    def forward(self, src_tokens, **kwargs):
        device = src_tokens.device
        batch_size, t = src_tokens.size()
        
        global_bytes_embedded, idx_local, masks = self.prepare_input(src_tokens, kwargs['target_acoustic_mask'])
        global_in = rearrange(global_bytes_embedded, "b (t p) e -> b t (p e)", p=self.patch_size) # cat along p

        prompt_encoded = self.ln(kwargs['text_feature'])
                
        global_in[:, 3: 3 + self.prompt_latent_length] = prompt_encoded
        
        pos = torch.arange(0, t // self.patch_size, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.global_model.transformer.wpe(pos)
        global_in += pos_emb
        
        masks = masks[:, ::self.patch_size]
        
        global_output = self.global_model(global_in, masks)
        global_output_reshaped = rearrange(global_output, "b t (p e) -> (b t) p e", p=self.patch_size) # reshape as B*t, p, e
        global_output_reshaped = self.linear_map(global_output_reshaped)

        # Local
        local_bytes_embedded = self.local_model.get_embeding(idx_local) # B, p, e
        local_in = local_bytes_embedded + global_output_reshaped # B, p, e
        local_output = self.local_model(local_in) # 
                
        x = rearrange(local_output, "(b t) l v -> b (t l) v", b=batch_size) # b, len, v
        logits = self.lm_head(x) #

        return logits, x
    

    def get_normalized_probs(
        self,
        net_output,
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
    
    
    def prepare_input(self, idx, masks):
        padding_global = idx.new(idx.shape[0], self.patch_size).fill_(self.pad) # using pad to fill
        bytes_global = torch.cat((padding_global, idx[:, : -self.patch_size]), -1) # set a padding to the first patch
        bytes_input = rearrange(idx, "b (t p) -> (b t) p", p=self.patch_size) 
        padding_local = bytes_input.new(bytes_input.shape[0], 1).fill_(self.pad)
        bytes_local = torch.cat((padding_local, bytes_input[:, :-1]), -1) # (b*t, pad + p)

        masks = torch.cat([
                torch.zeros_like(padding_global).bool(),
                masks[:, :-self.patch_size]
            ], dim=-1)
        
        global_bytes_embedded = self.global_model.transformer.wte(bytes_global)  # b,(t p) e, should to add postional embedding
        
        return global_bytes_embedded, bytes_local, masks
    

    def compute_accuracy(self, lprobs, target):
        n_correct = torch.sum(
            lprobs.argmax(-1).eq(target)
        )
        total = torch.sum(target.ne(-1))
        return n_correct / total
    
    
    @torch.no_grad()
    def generate_bias(self, idx, text_feature, max_new_tokens, min_new_tokens, f0_start_index, f0_end_index, codebook_start_index, codebook_end_index, f0_end_token, acoustic_start_token, acoustic_end_token, temperature=0.8, top_k=None):
        device = idx.device
        batch_size, st_idx = idx.shape

        ans = [[] for i in range(self.patch_size)]
        codebook_length = 1024

        is_last_token  = False

        text_feature = self.ln(torch.tensor(text_feature, dtype=torch.float16, device=device))

        for i in tqdm(range(max_new_tokens)):
            if is_last_token:
                break

            is_acoustic_interval = i > 2
            
            padding_global = idx.new(idx.shape[0], self.patch_size).fill_(self.pad) # using pad to fill
            idx_global = torch.cat((padding_global, idx), -1) # set a padding to the first patch, 
            
            global_bytes_embedded = self.global_model.transformer.wte(idx_global)
            global_in = rearrange(global_bytes_embedded, "b (t p) e -> b t (p e)", p=self.patch_size)  # [B, 501, 256*3]
            
            global_in[:, 3:3 + self.prompt_latent_length] = text_feature # padding, T5_start, continuous token

            t = idx_global.shape[1]
            pos = torch.arange(0, t // 3, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
            pos_emb = self.global_model.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
            global_in += pos_emb
            
            global_output = self.global_model(global_in) # 
            global_output_reshaped = rearrange(global_output, "b t (p e) -> (b t) p e", p=self.patch_size) # reshape as B, p, e
            global_output_reshaped = self.linear_map(global_output_reshaped) # 
            padding_one_patch = idx.new(idx.shape[0], self.patch_size).fill_(self.pad) # 1,path_size
            idx = torch.cat([idx, padding_one_patch], dim=1) # 1,len + patch_size
            for j in range(self.patch_size):

                if not is_acoustic_interval and j > 0:
                    idx[:,st_idx] = cat_idx_next # update it
                    st_idx += 1  # move it
                    ans[j].append(cat_idx_next[0,0])
                    continue

                if i == 1 and j == 0:
                    cat_idx_next = torch.ones_like(idx_next) * f0_end_token
                    idx[:,st_idx] = cat_idx_next # update it
                    st_idx += 1  # move it
                    ans[j].append(cat_idx_next[0,0])
                    continue

                elif i == 2 and j == 0:
                    cat_idx_next = torch.ones_like(idx_next) * acoustic_start_token
                    idx[:,st_idx] = cat_idx_next # update it
                    st_idx += 1  # move it
                    ans[j].append(cat_idx_next[0,0])
                    continue

                elif i == 0 and j == 0:
                    logit_start_idx = f0_start_index
                    logit_end_idx = f0_end_index

                else:
                    logit_start_idx = codebook_start_index + j * 1024
                    logit_end_idx = codebook_start_index + (j+1) * 1024

                bytes_input = rearrange(idx, "b (t p) -> (b t) p", p=self.patch_size) # 
                padding_local = bytes_input.new(bytes_input.shape[0], 1).fill_(self.pad)
                idx_local = torch.cat((padding_local, bytes_input[:, :-1]), -1) # (b*t, pad + p)
                local_bytes_embedded = self.local_model.get_embeding(idx_local) # B, p, e
                local_in = local_bytes_embedded + global_output_reshaped # B, p, e
                local_output = self.local_model(local_in) # 
                x = rearrange(local_output, "(b t) l v -> b (t l) v", b=batch_size) # b, len, v
                logits = self.lm_head(x) #
                current_logits = logits[:,st_idx,:] / temperature # 1, dim
                relevant_logits = current_logits[:, logit_start_idx:logit_end_idx]  # only care about part

                if i > min_new_tokens and i > 2 and j==0: # 当采样到min_new_tokens，以及是第一个codebook,才考虑采样stop token
                    stop_logits = logits[:, st_idx,acoustic_end_token:acoustic_end_token+1]
                    relevant_logits = torch.cat([relevant_logits, stop_logits], dim=1) # we put the stop token to the last position
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(relevant_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                if idx_next[0,0] >= codebook_length:
                    is_last_token = True
                    break
                
                cat_idx_next = idx_next +  logit_start_idx # transfer to real
                idx[:,st_idx] = cat_idx_next # update it
                st_idx += 1  # move it
                ans[j].append(cat_idx_next[0,0])

        ans = torch.tensor(ans)
        return ans


@register_model_architecture(model_name="acoustic_lm", arch_name="acoustic_lm_tiny_nonprefix")
def base_architecture_noprefix(args):
    args.n_state = safe_getattr(args, "n_state", 576)
    args.n_head = safe_getattr(args, "n_head", 8)
    args.n_layer = safe_getattr(args, "n_layer", 12)
    args.prefix_lm = safe_getattr(args, "prefix_lm", False)
    args.dropout = safe_getattr(args, "dropout", 0.0)


@register_model_architecture(model_name="acoustic_lm", arch_name="acoustic_lm_global300M_noprefix")
def architecture_global300M_noprefix(args):
    args.n_state = safe_getattr(args, "n_state", 1152)
    args.n_head = safe_getattr(args, "n_head", 16)
    args.n_layer = safe_getattr(args, "n_layer", 20)
    args.prefix_lm = safe_getattr(args, "prefix_lm", False)
    args.dropout = safe_getattr(args, "dropout", 0.0)

