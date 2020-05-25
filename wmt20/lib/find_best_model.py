import os
import glob
import re
import numpy as np
import argparse
from wmt20.lib.io_helper import load_txt_file, write_lines, run_cmd

src_mt_txt = '/data/wangke/wmt20/data/raw/2020/dev/src_mt.txt'
pe_txt = '/data/wangke/wmt20/data/raw/2020/dev/pe.txt'
tercom_shell = 'java -jar ../OpenNMT-tf/APEplusQE/3rd_party/tercom-0.7.25/tercom.7.25.jar'
bleu_shell = 'tools/multi-bleu.perl'


def compute_BLEU(mt_file, pe_file=pe_txt):
    result_tmp_file = mt_file + '.bleu.tmp'
    cmd = '%s %s < %s | head -n 1 >%s' % (bleu_shell, pe_file, mt_file, result_tmp_file)
    try:
        run_cmd(cmd, throw_error=True)
        with open(result_tmp_file, 'r', encoding='utf8') as f:
            s = f.read()
        bleu = float(re.match('BLEU = [0-9\\.]+', s).group().split('=')[1])
    except Exception as e:
        print(e)
        print('Error occurred when calculate BLEU for %s and %s' % (mt_file, pe_file))
        bleu = 0.0
    os.system('rm -rf %s' % result_tmp_file)
    return bleu


def compute_ter_with_shift(mt_file, pe_file=pe_txt):
    mts = load_txt_file(mt_file, False)
    mts = ['%s (A-%d)' % (x.strip(), i) for i, x in enumerate(mts)]
    tmp_mt_file = mt_file + '.num'
    write_lines(mts, tmp_mt_file)
    pes = load_txt_file(pe_file, False)
    pes = ['%s (A-%d)' % (x.strip(), i) for i, x in enumerate(pes)]
    tmp_pe_file = pe_file + '.num'
    write_lines(pes, tmp_pe_file)
    tmp_ter_file = mt_file + '.ter.output.tmp'
    cmd = '%s -s -h %s -r %s > %s' % (tercom_shell, tmp_mt_file, tmp_pe_file, tmp_ter_file)
    run_cmd(cmd, throw_error=True)
    lines = load_txt_file(tmp_ter_file, False)

    ter = None
    for line in lines:
        if line.startswith('Total TER:'):
            ter = float(line.split('(')[0].split(':')[1])
            break
    if ter is None:
        ter = 1.0
    os.remove(tmp_mt_file)
    os.remove(tmp_pe_file)
    os.remove(tmp_ter_file)
    return ter


def inference(ckpt, override=False, **kwargs):
    model_dir = os.path.dirname(ckpt)
    eval_dir = os.path.join(model_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    step = int(re.search('step_[0-9]+', os.path.basename(ckpt)).group().split('_')[1])
    params = {
        'model': ckpt,
        'src': src_mt_txt,
        'output': os.path.join(eval_dir, 'predictions.txt.%d' % step),
        'gpu': -1,
        'beam_size': 4,
        'min_length': 2,
        'batch_size': 64,
        'length_penalty': 'avg'
    }
    params.update(kwargs)
    result_file = params['output']
    if not override and os.path.exists(result_file):
        return result_file
    cmd = 'python translate.py ' + ' '.join(['-%s %s' % (k, str(v)) for k, v in params.items()])
    run_cmd(cmd, throw_error=True)
    return result_file


def recover_tokens(in_file, out_file=None, override=False):
    if out_file is None:
        out_file = in_file + '.tok'
    if not override and os.path.exists(out_file):
        return out_file
    cmd = "cat %s | sed 's/ \\#\\#//g' > %s" % (in_file, out_file)
    run_cmd(cmd, throw_error=True)
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()
    ckpts = glob.glob('%s/*step_*.pt' % args.model_dir)
    ape_subwords = [inference(ckpt, override=False, gpu=args.gpu) for ckpt in ckpts]
    ape_txts = [recover_tokens(x, override=True) for x in ape_subwords]
    bleus = [compute_BLEU(x) for x in ape_txts]
    ters = [compute_ter_with_shift(x) for x in ape_txts]
    max_bleu_index = int(np.argmax(bleus))
    min_ter_index = int(np.argmin(ters))
    for ape_txt, bleu, ter in zip(ape_txts, bleus, ters):
        print('TER: %f, BLEU: %f, file: %s' % (ter, bleu, ape_txt))
    print('\n min TER: %f, file: %s' % (ters[min_ter_index], ape_txts[min_ter_index]))
    print('max BLEU: %f, file: %s' % (bleus[max_bleu_index], ape_txts[max_bleu_index]))


if __name__ == '__main__':
    main()
