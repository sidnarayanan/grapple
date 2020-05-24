#!/usr/bin/env python3

from grapple.utils import * 

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt 
import os
from tqdm import tqdm

from loguru import logger

parser = ArgumentParser()
parser.add_args(
    ('--infiles', {'nargs': '+', 'required': True}),
    ('--plot', {'required': True}),
)
args = parser.parse_args()

if __name__ == '__main__':

    for f in args.infiles:
        key, f = f.split('=')
        data = pkl.load(open(f, 'rb'))
        fp, tp = data['fp'], data['tp']
        auc = np.trapz(tp, x=fp)
        plt.plot(fp, tp, label=f'{key} ({auc:.3f})')

    # plt.yscale('log')
    # plt.xscale('log')
    plt.ylim(bottom=0.01)
    plt.xlim(left=0.01)

    plt.ylabel('True Neutral Positive Rate')
    plt.xlabel('False Neutral Positive Rate')
    plt.legend()

    for ext in ('pdf', 'png'):
        plt.savefig(args.plot + '.' + ext)

