import torch
from torch import nn 
from .utils import t2n 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

import pyjet
from pyjet.testdata import get_event

from loguru import logger

EPS = 1e-4


class Metrics(object):
    def __init__(self, device, softmax=True):
        self.loss_calc = nn.CrossEntropyLoss(
                ignore_index=-1, 
                reduction='none'
                # weight=torch.FloatTensor([1, 5]).to(device)
            )
        self.reset()
        self.apply_softmax = softmax

    def reset(self):
        self.loss = 0 
        self.acc = 0 
        self.pos_acc = 0
        self.neg_acc = 0
        self.n_pos = 0
        self.n_particles = 0
        self.n_steps = 0
        self.hists = {}

    @staticmethod
    def make_roc(pos_hist, neg_hist):
        pos_hist = pos_hist / pos_hist.sum()
        neg_hist = neg_hist / neg_hist.sum()
        tp, fp = [], []
        for i in np.arange(pos_hist.shape[0], -1, -1):
            tp.append(pos_hist[i:].sum())
            fp.append(neg_hist[i:].sum())
        auc = np.trapz(tp, x=fp)
        plt.plot(fp, tp, label=f'AUC={auc:.3f}')
        return fp, tp

    def add_values(self, yhat, y, label, idx, w=None):
        if w is not None:
            w = w[y==label]
        hist, self.bins = np.histogram(yhat[y==label], bins=np.linspace(0, 1, 100), weights=w)
        if idx not in self.hists:
            self.hists[idx] = hist + EPS
        else:
            self.hists[idx] += hist

    def compute(self, yhat, y, orig_y, w=None, m=None):
        # yhat = [batch, particles, labels]; y = [batch, particles]
        loss = self.loss_calc(yhat.view(-1, yhat.shape[-1]), y.view(-1))
        if w is not None:
            wv = w.view(-1)
            loss *= wv
        if m is None:
            m = np.ones_like(t2n(y), dtype=bool)
        loss = torch.mean(loss)
        self.loss += t2n(loss).mean()

        mask = (y != -1)
        n_particles = t2n(mask.sum())

        pred = torch.argmax(yhat, dim=-1) # [batch, particles]
        pred = t2n(pred)
        y = t2n(y)
        mask = t2n(mask)

        acc = (pred == y)[mask].sum() / n_particles 
        self.acc += acc

        n_pos = np.logical_and(m, y == 1).sum()
        pos_acc = (pred == y)[np.logical_and(m, y == 1)].sum() / n_pos
        self.pos_acc += pos_acc
        neg_acc = (pred == y)[np.logical_and(m, y == 0)].sum() / (n_particles - n_pos)
        self.neg_acc += neg_acc

        self.n_pos += n_pos
        self.n_particles += n_particles

        self.n_steps += 1

        if self.apply_softmax:
            yhat = t2n(nn.functional.softmax(yhat, dim=-1))
        else:
            yhat = t2n(yhat)
        if w is not None:
            w = t2n(w).reshape(orig_y.shape)
            wm = w[m]
            wnm = w[~m]
        else:
            wm = wnm = None
        self.add_values(yhat[:,:,1][m], orig_y[m], 0, 0, wm)
        self.add_values(yhat[:,:,1][m], orig_y[m], 1, 1, wm)
        self.add_values(yhat[:,:,1][~m], orig_y[~m], 0, 2, wnm)
        self.add_values(yhat[:,:,1][~m], orig_y[~m], 1, 3, wnm)

        if self.n_steps % 50 == 0 and False:
            logger.info(t2n(y[0])[:10])
            logger.info(t2n(pred[0])[:10])
            logger.info(t2n(yhat[0])[:10, :])

        return loss, acc

    def mean(self):
        return ([x / self.n_steps 
                 for x in [self.loss, self.acc, self.pos_acc, self.neg_acc]]
                + [self.n_pos / self.n_particles])

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,
                'bins': self.bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists[0], label='PU Neutral', **hist_args)
        plt.hist(weights=self.hists[1], label='Hard Neutral', **hist_args)
        plt.hist(weights=self.hists[2], label='PU Charged', **hist_args)
        plt.hist(weights=self.hists[3], label='Hard Charged', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel('P(Hard|p,e)')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        plt.clf()
        fig_handle = plt.figure()
        fp, tp, = self.make_roc(self.hists[1], self.hists[0])
        plt.ylabel('True Neutral Positive Rate')
        plt.xlabel('False Neutral Positive Rate')
        path += '_roc'
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)
        pickle.dump({'fp': fp, 'tp': tp}, open(path + '.pkl', 'wb'))
        plt.close(fig_handle)


class METMetrics(Metrics):
    def __init__(self, device, softmax=True):
        super().__init__(device, softmax)

        self.mse = nn.MSELoss()
        self.met_loss_weight = 1

    def compute(self, yhat, y, orig_y, met, methat, w=None, m=None):
        met_loss = self.met_loss_weight * self.mse(methat.view(-1), met.view(-1))
        pu_loss, acc = super().compute(yhat, y, orig_y, w, m)
        loss = met_loss + pu_loss 
        return loss, acc

