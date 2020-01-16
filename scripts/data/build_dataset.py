#!/usr/bin/env python3

from grapple.data.cern import Event, Record, Grid
from grapple.utils import * 

import numpy as np
from tqdm import trange


parser = ArgumentParser()
parser.add_args(
    '--minbias', '--hard', '--output',
    ('--npu', {'type': int}),
    ('--nmax', {'type': int}),
    ('--nperfile', {'type': int}),
)
args = parser.parse_args()


def saveto(es, path):
    getidx = lambda i, es=es : [e[i] for e in es]

    x = np.stack(getidx(0), axis=0)
    y = np.stack(getidx(1), axis=0)
    N = np.array(getidx(2))

    np.savez(path, x=x, N=N, y=y)


if __name__ == '__main__':
    hard_rec = Record(args.hard)
    mb_rec = Record(args.minbias)
    grid = Grid(args.npu + 1)

    snapshot = Snapshot(args.output, args)
    file_idx = 0
    es = []
    for _ in trange(args.nmax):
        e = Event(hard_rec, mb_rec, args.npu, grid)
        es.append((e.x, e.y, e.N))
        if len(es) == args.nperfile:
            saveto(es, snapshot.get_path(f'data_{file_idx}.npz'))
            file_idx += 1
            es = []
    if es:
        saveto(es, snapshot.get_path(f'data_{len(es)}.npz'))
