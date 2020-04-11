import torch
from torch import nn 
from .utils import t2n 
import matplotlib.pyplot as plt
import numpy as np


class Metrics(object):
    def __init__(self, device):
        self.loss_calc = nn.CrossEntropyLoss(
                ignore_index=-1, 
                reduction='none'
                # weight=torch.FloatTensor([1, 5]).to(device)
            )
        self.reset()

    def reset(self):
        self.loss = 0 
        self.acc = 0 
        self.pos_acc = 0
        self.neg_acc = 0
        self.n_pos = 0
        self.n_particles = 0
        self.n_steps = 0

    def compute(self, yhat, y, w=None):
        # yhat = [batch, particles, labels]; y = [batch, particles]
        loss = self.loss_calc(yhat.view(-1, yhat.shape[-1]), y.view(-1))
        if w is not None:
            w = w.view(-1)
            loss *= w
        loss = torch.mean(loss)
        self.loss += t2n(loss).mean()

        mask = (y != -1)
        n_particles = t2n(mask.sum())

        pred = torch.argmax(yhat, dim=-1) # [batch, particles]
        acc = t2n((pred == y)[mask].sum()) / n_particles 
        self.acc += acc

        n_pos = t2n((y == 1).sum())
        pos_acc = t2n((pred == y)[y == 1].sum()) / n_pos
        self.pos_acc += pos_acc
        neg_acc = t2n((pred == y)[y == 0].sum()) / (n_particles - n_pos)
        self.neg_acc += neg_acc

        self.n_pos += n_pos
        self.n_particles += n_particles

        self.n_steps += 1

        if self.n_steps % 50 == 0 and False:
            print(t2n(y[0])[:10])
            print(t2n(pred[0])[:10])
            print(t2n(yhat[0])[:10, :])

        return loss, acc

    def mean(self):
        return ([x / self.n_steps 
                 for x in [self.loss, self.acc, self.pos_acc, self.neg_acc]]
                + [self.n_pos / self.n_particles])


class METResolution(object):
    def __init__(self, bins=np.linspace(-1, 2, 40)):
        self.bins = bins
        self.reset()

    def reset(self):
        self.dist = None
        self.dist_p = None

    @staticmethod
    def _compute_res(pt, phi, w, gm):
        pt = pt * w
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        metx = np.sum(px, axis=-1)
        mety = np.sum(py, axis=-1)
        met = np.sqrt(np.power(metx, 2) + np.power(mety, 2))
        res = (met / gm) - 1
        return res

    def compute(self, pt, phi, pf, gm, pred):
        res = (pred / gm) - 1
        res_p = (pf / gm) - 1

        hist, _ = np.histogram(res, bins=self.bins)
        hist_p, _ = np.histogram(res_p, bins=self.bins)
        if self.dist is None:
            self.dist = hist
            self.dist_p = hist_p
        else:
            self.dist += hist
            self.dist_p += hist_p

    @staticmethod 
    def _compute_moments(x, dist):
        dist = dist / np.sum(dist)
        mean = np.sum(x * dist) 
        var = np.sum(np.power(x - mean, 2) * dist) 
        return mean, var

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5
        plt.hist(x=x, weights=self.dist, label='Maierayanaschott', alpha=0.5, bins=self.bins)
        plt.hist(x=x, weights=self.dist_p, label='PF', alpha=0.5, bins=self.bins)
        plt.xlabel('(Predicted-True)/True')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        mean, var = self._compute_moments(x, self.dist)
        mean_p, var_p = self._compute_moments(x, self.dist_p)
        self.reset()

        return {'model': (mean, np.sqrt(var)), 'puppi': (mean_p, np.sqrt(var_p))}


class ParticleMETResolution(METResolution):
    @staticmethod
    def _compute_res(pt, phi, w, gm):
        pt = pt * w
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        metx = np.sum(px, axis=-1)
        mety = np.sum(py, axis=-1)
        met = np.sqrt(np.power(metx, 2) + np.power(mety, 2))
        res = (met / gm) - 1
        return res

    def compute(self, pt, phi, w, puppi, gm):
        res = self._compute_res(pt, phi, w, gm)
        res_p = self._compute_res(pt, phi, puppi, gm)

        hist, _ = np.histogram(res, bins=self.bins)
        hist_p, _ = np.histogram(res_p, bins=self.bins)
        if self.dist is None:
            self.dist = hist
            self.dist_p = hist_p
        else:
            self.dist += hist
            self.dist_p += hist_p
