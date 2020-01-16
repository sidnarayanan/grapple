import numpy as np


def cms_to_grapple(inpath, outpath):
    d = np.load(inpath)
    x = d['x']
    y = d['y']
    q = d['q']
    p = d['p']
    N = (x[:, :, 0] > 0).sum(axis=-1) 
    np.savez(outpath, x=x, y=y, N=N, q=q, p=p, met=d['met'])
