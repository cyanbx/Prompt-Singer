import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import os
import argparse
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from models import CodeBigVGAN as Generator
from tqdm import tqdm

h = None
device = None
torch.backends.cudnn.benchmark = False

MAX_WAV_VALUE = 32768.0

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def inference(a, h):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    input_file = open(a.input_code_file)
    input_lines = input_file.readlines()
    
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    
    syn_dir = os.path.join(a.output_dir, 'syn')  
    if not os.path.exists(syn_dir):  
        os.makedirs(syn_dir) 
        
    for i in tqdm(range(len(input_lines))):
        if input_lines[i][0] != 'D':
            continue
        generated_line = input_lines[i]

        item_name,  gen_code = generated_line.split('\t')
        item_name = item_name[2:]

        item_fname = item_name + '.wav'
        gen_fpath = os.path.join(a.output_dir, 'syn', item_fname) 
        
        audio_str = gen_code
        audio = audio_str.split()
        ori_audio = audio

        for i, x in enumerate(ori_audio):
            if (x == '<acoustic_start>' and ori_audio[i + 1] != '<acoustic_start>'):
                audio = audio[i + 1 : -4]
                break
        
        gen_code = [int (x[4:]) for x in audio if len(x) > 0]

        if len(gen_code) % 3 == 1:
            gen_code = gen_code[:-1]
        elif len(gen_code) % 3 == 2:
            gen_code = gen_code[:-2]
        
        for i, x in enumerate(gen_code):
            if gen_code[i] >= 2048:
                gen_code[i] -= 2048
                continue
            if gen_code[i] >= 1024:
                gen_code[i] -= 1024
                
        audio = torch.LongTensor(gen_code).reshape(-1, 3).transpose(0, 1).cuda()
        gen_code = np.array(gen_code)
        gen_code = torch.LongTensor(np.stack([gen_code[::3], gen_code[1::3], gen_code[2::3]])).unsqueeze(0).to(device)

        with torch.no_grad():
            y_g_hat = generator(gen_code)

        audio_g = y_g_hat.squeeze().detach() * MAX_WAV_VALUE
        audio_g = audio_g.cpu().numpy().astype('int16')

        write(gen_fpath, h.sampling_rate, audio_g)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_code_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--checkpoint_file', required=True)

    a = parser.parse_args()

    config_file = os.path.join(os.path.dirname(a.checkpoint_file), 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a, h)


if __name__ == '__main__':
    main()

