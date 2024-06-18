import os
import glob
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-tsv', type=str, required=True)
    args = parser.parse_args()

    items = glob.glob(f"{args.input_dir}/*.npy")
    output_file = open(args.output_tsv, 'w')
    output_file.write('item_name\tf0_avg\tf0_rescale\n')

    for e in tqdm(items):
        pitch = np.load(e)
        pitch_round = np.round(pitch).astype(int)
        pitch_clip = np.where(pitch_round > 850, 850, pitch_round)
        pitch_clip = np.where((pitch_clip > 0) & (pitch_clip < 45), 45, pitch_clip)
        pitch_nonzero = np.array([x for x in pitch_clip if x != 0])

        avg_value = np.round(np.mean(pitch_nonzero)).astype(int)
        rescale_value = np.round(pitch_clip * 222 / avg_value).astype(int)

        output_file.write(f"{os.path.basename(e)[:-4]}\t{avg_value}\t{' '.join([str(x) for x in rescale_value])}\n")
