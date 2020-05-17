#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko


import os
import re
import h5py
import random
from xml.etree import ElementTree

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


ascii_trans_pattern = re.compile('CSR: ?\n\n(.+)\n', re.DOTALL)
lines_path = 'dataset/lineStrokes-all/lineStrokes'
trans_path = 'dataset/ascii-all/ascii'


def get_dir_id(path):
    slices = path.split('/')
    return '/'.join(slices[-2:])


def process_ascii_dir(dir_id):
    """Processes ascii translations"""
    dir_path = os.path.join(trans_path, dir_id)
    file_name = os.listdir(dir_path)[0]
    file_path = os.path.join(dir_path, file_name)

    with open(file_path) as f:
        data = f.read()
        data = ascii_trans_pattern.search(data)
        return data.group(1).split('\n') if data else []


def normalize_line_points(line_points: np.ndarray):
    """Shift horizontally, normalize"""

    # we dont subtract mean horizontally as we want
    # to preserve 'move' of handwriting, otherwise we would get "fluctuations"
    # near zero
    # line_points[:, 0] -= np.min(line_points[:, 0])
    # line_points[:, 1] -= np.mean(line_points[:, 1])

    std = np.std(line_points[:, :2])
    line_points[:, 0] /= std
    line_points[:, 1] /= std
    return line_points


def relativize_line_points(line_points: np.ndarray):
    """Rebuild all absolute coordinates to relative offsets"""
    points = np.array(line_points)
    length, vec_size = points.shape
    res = np.empty(shape=(length - 1, vec_size))

    # consecutive differences of coordinates (offsets)
    res[:, 0] = points[1:, 0] - points[:-1, 0]
    res[:, 1] = -(points[1:, 1] - points[:-1, 1])

    # end of stroke
    res[:, 2] = points[1:, 2]
    return res


def process_line_points(line_points: list) -> np.ndarray:
    pts = np.array(line_points).astype(np.float16)
    pts = relativize_line_points(pts)
    pts = normalize_line_points(pts)
    return pts


def process_xml_file(path):
    """Process xml file with data about one line"""
    tree = ElementTree.parse(path)
    stroke_set = tree.find('StrokeSet')
    strokes = stroke_set.findall('Stroke')
    points = [s.findall('Point') for s in strokes]

    res_points = []
    for p_list in points:
        p_list = [[int(x.attrib['x']), int(x.attrib['y']), 0]
                  for x in p_list]
        p_list[-1][2] = 1
        res_points.extend(p_list)
    res_points = process_line_points(res_points)
    return res_points


def process_xml_dir(path):
    """Processed xml directory into array of points, separated by lines"""
    try:
        _, _, file_names = next(os.walk(path))
    except StopIteration: return []
    lines = []
    for file_name in file_names:
        xml_path = os.path.join(path, file_name)
        lines.append(process_xml_file(xml_path))
    return lines


def is_eligible(points, translations):
    """Implements all checks to avoid garbage collected into dataset"""
    checks = []
    non_empty = bool(points) and bool(translations)
    checks.append(non_empty)
    return all(checks)


def print_progressbar(amt, total_amt, status_msg='Processing dirs ', length=10):
    part = int((amt/total_amt) * length) - 1
    seq = '[' + '=' * part + '>' + '.' * (length - part) + ']'
    print(f'\r{status_msg}[{amt}/{total_amt}]\t{seq}', end='')


def main():
    lines = []
    translations = []

    for cur_dir, sub_dirs, files in os.walk(lines_path):
        if not files: continue
        line_points = process_xml_dir(cur_dir)
        dir_id = get_dir_id(cur_dir)
        translation_lines = process_ascii_dir(dir_id)

        if is_eligible(line_points, translation_lines):
            lines.extend(line_points)
            translations.extend(translation_lines)

    return lines, translations


def prepare(line):
    start = np.array([0., 0., 1.])
    # drop last element to preserve shape
    res = np.empty(line.shape)
    res[0] = start
    res[1:] = line[:-1]
    return res


def make_equal_lengths(dataset, margin=300):
    """

    Each line with less points then `margin` is dropped, each greater
    is replaced by rule: "pick random continuous sample of size `margin`
    len(target)//len(margin) times and append". According to
    https://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/
    """
    res = []
    for line in dataset:
        line_len = len(line)
        if line_len < margin:
            continue
        if line_len == margin:
            res.append(prepare(line))
        else:
            times = line_len // margin
            for i in range(times):
                start = random.randrange(0, line_len - margin)
                sample = line[start: start + margin]
                res.append(prepare(sample))
    return res


# batch_size: 1053
if __name__ == '__main__':
    import pickle
    # lines, translations = main()
    with open('data/lines.pkl', 'rb') as f:
        lines = pickle.load(f)
    lines = make_equal_lengths(lines)
    # lines = pad_sequences(lines, value=[0., 0., 0.],
    #                       padding='pre', truncating='pre',
    #                       maxlen=750, dtype='float32')

    # translations = np.array(translations)

    with h5py.File('dataset.h5', 'w') as f:
        f.create_dataset('lines', data=lines)
        # f.create_dataset('translations', data=translations)
