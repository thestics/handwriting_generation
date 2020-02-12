#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko


import os
import re
import pickle
from xml.etree import ElementTree

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


def relativize_line_points(line_points):
    """Rebuild all absolute coordinates to relative offsets"""
    res = []

    first = True
    cur_x, cur_y, _ = line_points[0]

    for point in line_points:
        if first:
            first = False
            res.append([0, 0, 0])
            continue
        x, y, end_status = point
        res.append([x - cur_x, y - cur_y, end_status])
        cur_x, cur_y = x, y
    return res


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
    res_points = relativize_line_points(res_points)
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

    amt = 0
    total_amt = 1711
    for cur_dir, sub_dirs, files in os.walk(lines_path):
        if not files: continue
        print_progressbar(amt, total_amt, length=20)
        # print(f'\rProcessing dir: {cur_dir}, amt: {amt}', end='')
        amt += 1
        line_points = process_xml_dir(cur_dir)
        dir_id = get_dir_id(cur_dir)
        translation_lines = process_ascii_dir(dir_id)

        if is_eligible(line_points, translation_lines):
            lines.extend(line_points)
            translations.extend(translation_lines)

        if amt > 100: break

    return lines, translations


# batch_size: 1053
if __name__ == '__main__':
    lines, translations = main()
    max_line = len(max(lines, key=lambda row: len(row)))
    upd_lines = []
    for line in lines:
        new_line = [[0, 0, 0] for i in range(max_line - len(line))] + line
        upd_lines.append(new_line)
    lines = np.array(upd_lines)

    translations = np.array(translations)

    np.save('lines.npy', lines, allow_pickle=True)
    np.save('translations.npy', translations, allow_pickle=True)
