import os
import numpy as np


def load_txt_file(file, convert2array=True):
    with open(file, 'r', encoding='utf8') as f:
        lines = [x.strip() for x in f.readlines()]
    if convert2array:
        lines = np.array(lines)
    return lines


def load_tokens_from_file(file, convert2array=True):
    lines = load_txt_file(file, convert2array=convert2array)
    if convert2array:
        return [np.array(x.split(), dtype=str) for x in lines]
    else:
        return [x.split() for x in lines]


def load_tags_from_file(file, convert2array=True):
    lines = load_txt_file(file)
    if convert2array:
        return [np.array(line.split()).astype(int) for line in lines]
    else:
        return [list(map(int, line.split())) for line in lines]


def write_tokens_to_file(tokens, file):
    with open(file, 'w', encoding='utf8') as f:
        f.write('\n'.join([' '.join(x) for x in tokens]))


def write_lines(lines, file):
    with open(file, 'w', encoding='utf8') as f:
        f.write('\n'.join([str(x).strip() for x in lines]))


def generate_odps_table(in_file, out_file, src_lang='en', tgt_lang='de', delimiter='\005', id_prefix=None):
    if id_prefix is None:
        id_prefix = in_file
    src = load_txt_file(in_file, False)
    num = len(src)
    ids = ['%s_%d' % (id_prefix, x) for x in range(num)]
    src_lang = [src_lang] * num
    tgt_lang = [tgt_lang] * num
    table = zip(ids, src_lang, tgt_lang, src)
    with open(out_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([delimiter.join(x) for x in table]))


def run_cmd(cmd, throw_error=False):
    return_code = os.system(cmd)
    if return_code != 0:
        error_log = 'Error occurred while run "%s", return code: %d' % (cmd, return_code)
        if throw_error:
            raise RuntimeError(error_log)
        else:
            print(error_log)
