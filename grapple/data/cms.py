import numpy as np
from loguru import logger


def cms_to_grapple(inpath, outpath):
    logger.info(f'Loading from {inpath}')
    d = np.load(inpath)
    x = d['x']
    y = d['y']
    q = d['q']
    p = d['p']
    N = (x[:, :, 0] > 0).sum(axis=-1) 
    logger.info(f'Writing to {outpath}')
    np.savez(outpath, x=x, y=y, N=N, q=q, p=p, met=d['met'])
