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
            mask_base = np.arange(X.shape[1])
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            for i in idx:
                mask = (mask_base < N[i]).astype(int) 
                y = (Y[i, :] == 0).astype(int)
                y[~mask] = -1
                x = X[i, :, :]
                if self.mask_charged:
                    q_mask = (x[:, -1] > -1)
                    y[q_mask] = -1
                yield x, y, mask 

    @staticmethod
    def collate_fn(samples):
        X = np.stack([s[0] for s in samples], axis=0)
        Y = np.stack([s[1] for s in samples], axis=0)
        M = np.stack([s[2] for s in samples], axis=0)

        return X, Y, M

