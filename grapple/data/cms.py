import numpy as np
from loguru import logger


def cms_to_grapple(inpath, outpath):
    logger.info(f'Loading from {inpath}')
    d = np.load(inpath)
    N = (d['x'][:, :, 0] > 0).sum(axis=-1) 
    kwargs = {k:d[k] for k in d.keys()}
    kwargs['N'] = N
    logger.info(f'Writing to {outpath}')
    np.savez(outpath, **kwargs)
