import warnings
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import re
import json
import torch
from collections import OrderedDict
import csv
import numpy as np
from tqdm import tqdm  
import argparse


def is_sil_phoneme(p):
    return p == 'SP'

def remove_empty_lines(text):
    """remove empty lines"""
    assert (len(text) > 0)
    assert (isinstance(text, list))
    text = [t.strip() for t in text]
    if "" in text:
        text.remove("")
    return text

class TextGrid(object):
    def __init__(self, text):
        text = remove_empty_lines(text)
        self.text = text
        self.line_count = 0
        self._get_type()
        self._get_time_intval()
        self._get_size()
        self.tier_list = []
        self._get_item_list()

    def _extract_pattern(self, pattern, inc):
        """
        Parameters
        ----------
        pattern : regex to extract pattern
        inc : increment of line count after extraction
        Returns
        -------
        group : extracted info
        """
        try:
            group = re.match(pattern, self.text[self.line_count]).group(1)
            self.line_count += inc
        except AttributeError:
            raise ValueError("File format error at line %d:%s" % (self.line_count, self.text[self.line_count]))
        return group

    def _get_type(self):
        self.file_type = self._extract_pattern(r"File type = \"(.*)\"", 2)

    def _get_time_intval(self):
        self.xmin = self._extract_pattern(r"xmin = (.*)", 1)
        self.xmax = self._extract_pattern(r"xmax = (.*)", 2)

    def _get_size(self):
        self.size = int(self._extract_pattern(r"size = (.*)", 2))

    def _get_item_list(self):
        """Only supports IntervalTier currently"""
        for itemIdx in range(1, self.size + 1):
            tier = OrderedDict()
            item_list = []
            tier_idx = self._extract_pattern(r"item \[(.*)\]:", 1)
            tier_class = self._extract_pattern(r"class = \"(.*)\"", 1)
            if tier_class != "IntervalTier":
                raise NotImplementedError("Only IntervalTier class is supported currently")
            tier_name = self._extract_pattern(r"name = \"(.*)\"", 1)
            tier_xmin = self._extract_pattern(r"xmin = (.*)", 1)
            tier_xmax = self._extract_pattern(r"xmax = (.*)", 1)
            tier_size = self._extract_pattern(r"intervals: size = (.*)", 1)
            for i in range(int(tier_size)):
                item = OrderedDict()
                item["idx"] = self._extract_pattern(r"intervals \[(.*)\]", 1)
                item["xmin"] = self._extract_pattern(r"xmin = (.*)", 1)
                item["xmax"] = self._extract_pattern(r"xmax = (.*)", 1)
                item["text"] = self._extract_pattern(r"text = \"(.*)\"", 1)
                item_list.append(item)
            tier["idx"] = tier_idx
            tier["class"] = tier_class
            tier["name"] = tier_name
            tier["xmin"] = tier_xmin
            tier["xmax"] = tier_xmax
            tier["size"] = tier_size
            tier["items"] = item_list
            self.tier_list.append(tier)

    def toJson(self):
        _json = OrderedDict()
        _json["file_type"] = self.file_type
        _json["xmin"] = self.xmin
        _json["xmax"] = self.xmax
        _json["size"] = self.size
        _json["tiers"] = self.tier_list
        return json.dumps(_json, ensure_ascii=False, indent=2)


def get_mel2ph_sing(tg_fn, item, trim_bos_eos=False):
    with open(tg_fn, "r") as f:
        tg = f.readlines()
    tg = remove_empty_lines(tg)
    tg = TextGrid(tg)
    tg = json.loads(tg.toJson())
    tg_idx = 0
    ph_idx = 0
    tg_align = [x for x in tg['tiers'][-1]['items']]
    tg_align_ = []
    ph_list = []
    for x in tg_align:
        x['xmin'] = float(x['xmin'])
        x['xmax'] = float(x['xmax'])
        if x['text'] in ['sil', 'sp', '', 'SIL', 'PUNC', '<SP>', 'SP','|','#']:
            if len(ph_list) == 0 or ph_list[-1] != 'SP':
                ph_list.append('SP')
            x['text'] = ''
            if len(tg_align_) > 0 and tg_align_[-1]['text'] == '':
                tg_align_[-1]['xmax'] = x['xmax']
                continue
        elif x['text'] in ['AP', '<AP>']:
            x['text'] = 'AP'
            ph_list.append('AP')
        else:
            ph_list.append(x['text'])

        if not (len(tg_align_) == 0 and x['text'] == ''):
            tg_align_.append(x)

    split = np.ones(len(ph_list) + 1, float) * -1
    tg_align = tg_align_
    tg_len = len([x for x in tg_align if x['text'] != ''])
    ph_len = len([x for x in ph_list if not is_sil_phoneme(x)])
    assert tg_len == ph_len, (tg_len, ph_len, tg_align, ph_list, tg_fn)
    while tg_idx < len(tg_align) or ph_idx < len(ph_list):
        if tg_idx == len(tg_align) and is_sil_phoneme(ph_list[ph_idx]):
            split[ph_idx] = 1e8
            ph_idx += 1
            continue
        x = tg_align[tg_idx]
        if x['text'] == '' and ph_idx == len(ph_list):
            tg_idx += 1
            continue
        assert ph_idx < len(ph_list), (tg_len, ph_len, tg_align, ph_list, tg_fn)
        ph = ph_list[ph_idx]
        if x['text'] == '' and not is_sil_phoneme(ph):
            assert False, (ph, ph_list, tg_align)
        if x['text'] != '' and is_sil_phoneme(ph):
            ph_idx += 1
        else:
            assert (x['text'] == '' and is_sil_phoneme(ph)) \
                   or re.sub(r'\d+', '', x['text'].lower()) == re.sub(r'\d+', '', ph.lower()) \
                   or x['text'].lower() == 'sil', (x['text'], ph, [x['text'] for x in tg_align], ph_list)
            split[ph_idx] = x['xmin']
            if ph_idx > 0 and split[ph_idx - 1] == -1 and is_sil_phoneme(ph_list[ph_idx - 1]):
                split[ph_idx - 1] = split[ph_idx]
            ph_idx += 1
            tg_idx += 1
    assert tg_idx == len(tg_align), (tg_idx, [x['text'] for x in tg_align])
    assert ph_idx >= len(ph_list) - 1, (ph_idx, ph_list, len(ph_list), [x['text'] for x in tg_align], tg_fn)
    mel2ph = np.zeros([int(item['n_frames'])], int)
    split[0] = 0
    split[-1] = 1e8
    for i in range(len(split) - 1):
        assert split[i] != -1 and split[i] <= split[i + 1], (split[:-1],)
    split = [int(s * 50 + 0.5) for s in split]
    for ph_idx in range(len(ph_list)):
        mel2ph[split[ph_idx]:split[ph_idx + 1]] = ph_idx + 1
    mel2ph_torch = torch.from_numpy(mel2ph)
    T_t = len(ph_list)
    dur = mel2ph_torch.new_zeros([T_t + 1]).scatter_add(0, mel2ph_torch, torch.ones_like(mel2ph_torch))
    dur = list(dur[1:].numpy())
    if trim_bos_eos:
        bos_dur = dur[0]
        eos_dur = dur[-1]
        dur = dur[1:-1]
        mel2ph = mel2ph[bos_dur:-eos_dur]
    return mel2ph, dur, ph_list

def get_expand_phone(tg_fn, item):
    l = int(item['n_frames'])
    mel2ph, durs, ph_list = get_mel2ph_sing(tg_fn, item)
    
    # if ph_list[0] in ['sil', 'sp', '', 'SIL', 'PUNC', '<SP>', 'SP','|','#']:
    #     ph_list[0] = "SP"
    
    # if ph_list[-1] in ['sil', 'sp', '', 'SIL', 'PUNC', '<SP>', 'SP','|','#']:
    #     ph_list[-1] = "SP"

    ph_expand = [ph_list[ph_idx - 1] for ph_idx in mel2ph]
    durs_, ph_ = [d for d in durs if d != 0], []
    start_frame = 0
    for d in durs_:
        ph_.append(ph_expand[start_frame])
        start_frame += d
    assert len(ph_expand) == sum(durs_) and len(ph_expand) == l, (len(ph_expand), sum(durs_), l)
    return ph_expand, " ".join(ph_), durs_

                 
def remove_digits_from_list(lst):  
    result = []  
    for item in lst:  
        item = re.sub(r'\d', '', item)  
        if item == '#' or item == '|':
            item = 'SP'
        result.append(item)  
    return result


def cnt_diff(ref, gen):
    diff = abs(len(ref) - len(gen))
    min_len = min(len(ref), len(gen))
    cnt = 0
    for i in range(min_len):
        if ref[i] != gen[i] and not (ref[i] == 'SP' and gen[i] == 'AP' or ref[i] == 'AP' and gen[i] == 'SP'):
            cnt += 1
    
    return cnt, diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=str, required=True)
    parser.add_argument("--output-tsv", type=str, required=True)
    parser.add_argument("--textgrid-dir", type=str, required=True)

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

    output_file = open(args.output_tsv, 'w')
    output_file.write('item_name\tphone\n')

    for i in tqdm(items):
        textgrid_path = os.path.join(args.textgrid_dir, i['item_name'] + '.TextGrid')
        if os.path.exists(textgrid_path):
            a,b,c = get_expand_phone(textgrid_path, i)
            a = remove_digits_from_list(a)
            output_file.write(f"{i['item_name']}\t{' '.join(a)}\n")
