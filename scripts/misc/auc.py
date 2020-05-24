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
    ('--plot', {'required': True}),
)
args = parser.parse_args()

if __name__ == '__main__':
    x = [0, 0.2, 0.4, 0.6, 1.0]
    y = [0.825, 0.865, 0.865, 0.862, 0.826] 
    plt.plot(x, y, label='Model')

    y = [0.784] * len(x)
    plt.plot(x, y, label='Puppi', linestyle='--')

    plt.legend()
    plt.ylabel('AUC')
    plt.xlabel(r'Maximum $\Delta R$')

    for ext in ('pdf', 'png'):
        plt.savefig(args.plot + '.' + ext)

