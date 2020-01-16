import torch
from torch.utils.data import DataLoader, IterableDataset
from loguru import logger
from glob import glob 
import numpy as np
from tqdm import tqdm


class PUDataset(IterableDataset):
    def __init__(self, config):
        self._files = glob(config.dataset_pattern)
        self.mask_charged = config.mask_charged

    def __iter__(self):
        np.random.shuffle(self._files)
        for f in self._files:
            data = np.load(f)
            X = data['x']
            Y = data['y']
            N = data['N']
            has_pq = 'p' in data
            if has_pq:
                P = data['p']
                Q = data['q']
                genmet = data['met']
            mask_base = np.arange(X.shape[1])
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            for i in idx[:400]:
                mask = (mask_base < N[i]).astype(int) 
                y = Y[i, :].astype(int)
                y[~mask] = -1
                x = X[i, :, :]
                if self.mask_charged:
                    q_mask = (x[:, -1] > -1)
                    y[q_mask] = -1
                to_yield = (x, y, mask)
                if has_pq:
                    to_yield += (Q[i, :], P[i, :], genmet[i])
                yield to_yield

    @staticmethod
    def collate_fn(samples):
        n_fts = len(samples[0])
        to_ret = [np.stack([s[i] for s in samples], axis=0) for i in range(n_fts)]
        return to_ret
