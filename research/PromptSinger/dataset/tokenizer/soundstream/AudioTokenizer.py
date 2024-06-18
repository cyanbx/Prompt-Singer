#

"""Command-line for audio compression."""
import os
import torch
from omegaconf import OmegaConf
import logging

from research.PromptSinger.dataset.tokenizer.abs_tokenizer import AbsTokenizer
from research.PromptSinger.dataset.tokenizer.soundstream.models.soundstream import SoundStream


class AudioTokenizer(AbsTokenizer):
    def __init__(self, 
                 ckpt_path,
                 device=torch.device('cpu'), 
                 ):
        """ soundstream with fixed bandwidth of 4kbps 
            It encodes audio with 50 fps and 8-dim vector for each frame
            The value of each entry is in [0, 1023]
        """
        super(AudioTokenizer, self).__init__()
        # GPU is only for offline tokenization
        # So, when distributed training is launched, this should still be on CPU

        self.device = device
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.yaml')
        if not os.path.isfile(config_path):
            raise ValueError(f"{config_path} file does not exist.")
        config = OmegaConf.load(config_path)
        
        self.ckpt_path = ckpt_path
        logging.info(f"using config {config_path} and model {self.ckpt_path}")
        
        self.soundstream = self.build_codec_model(config)
        # properties
        self.sr = 16000
        self.dim_codebook = 1024
        self.n_codebook = 3
        self.bw = 1.5 # bw=1.5 ---> 3 codebooks
        self.freq = self.n_codebook * 50
        self.mask_id = self.dim_codebook * self.n_codebook
        

    def build_codec_model(self, config):
        model = eval(config.generator.name)(**config.generator.config)
        parameter_dict = torch.load(self.ckpt_path, map_location='cpu')
        model.load_state_dict(parameter_dict['codec_model']) # load model
        model = model.to(self.device)
        return model
    
    
    @torch.no_grad()
    def encode(self, wav):
        wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
        compressed = self.soundstream.encode(wav, target_bw=self.bw) # [n_codebook, 1, n_frames]
        compressed = compressed.squeeze(1).detach().cpu().numpy() # [n_codebook, n_frames]

        return compressed
    

if __name__ == '__main__':
    tokenizer = AudioTokenizer(device=torch.device('cuda:0')).cuda()
    wav = '/home/v-dongyang/data/FSD/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/FreeSound_flac/537271.flac'
    codec = tokenizer.tokenize(wav)
    print(codec)

