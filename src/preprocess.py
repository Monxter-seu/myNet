#!/usr/bin/env python
# Created on 2018/12
# Author: Kaituo XU

# 文件夹命名格式为$num1_$num2

import argparse
import json
import os

import librosa


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    docu_list = os.listdir(in_dir)
    for docu in docu_list:
        docu_dir = os.path.join(in_dir, docu)
        csv_list = os.listdir(docu_dir)
        for csv_file in csv_list:
            if not csv_file.endswith('.csv'):
                continue
            csv_path = os.path.join(docu_dir, csv_file)
            label = [docu[0], docu[2]]
            file_infos.append((csv_path, label))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
            json.dump(file_infos, f, indent=4)


def preprocess(args):
    for data_type in ['tr', 'cv', 'tt']:
        for speaker in ['mix']:
            preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),
                               os.path.join(args.out_dir, data_type),
                               speaker,
                               sample_rate=args.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSJ0 data preprocessing")
    parser.add_argument('--in-dir', type=str, default=None,
                        help='Directory path of wsj0 including tr, cv and tt')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Directory path to put output files')
    parser.add_argument('--sample-rate', type=int, default=8000,
                        help='Sample rate of audio file')
    args = parser.parse_args()
    print(args)
    preprocess(args)
