import os
import csv
import numpy as np
import librosa
import pyworld as pw
from tqdm import tqdm
from argparse import ArgumentParser

def my_get_pitch(wav_path, sr=16000):
    wav_data, loaded_sr = librosa.core.load(wav_path, sr=sr)
    f0, _ = pw.harvest(wav_data.astype(np.double), sr, frame_period=20)
    return f0

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-tsv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    
    args = parser.parse_args()

    input_file = open(args.input_tsv)
    items = csv.DictReader(
        input_file,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )

    items = [i for i in items]

    for i in tqdm(items):
        f0 = my_get_pitch(i['audio_path'])
        out_path = os.path.join(args.output_dir, i['item_name'] + '.npy')
        np.save(out_path, f0)

