import torch
from torch import nn 
from .utils import t2n 
import matplotlib.pyplot as plt
import numpy as np
import pickle

import pyjet

EPS = 1e-12


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
            print(t2n(y[0])[:10])
            print(t2n(pred[0])[:10])
            print(t2n(yhat[0])[:10, :])

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


class JetResolution(object):
    def __init__(self):
        self.bins = {
                'pt': np.linspace(-100, 100, 100),
                'm': np.linspace(-50, 50, 100),
                'mjj': np.linspace(-1000, 1000, 100),
            }
        self.bins_2 = {
                'pt': np.linspace(0, 500, 100),
                'm': np.linspace(0, 50, 100),
                'mjj': np.linspace(0, 3000, 100),
            }
        self.labels = {
                'pt': 'Jet $p_T$',
                'm': 'Jet $m$',
                'mjj': '$m_{jj}$',
            }
        self.reset()

    def reset(self):
        self.dists = {k:[] for k in ['pt', 'm', 'mjj']} 
        self.dists_2 = {k:([], []) for k in ['pt', 'm', 'mjj']} 

    @staticmethod 
    def compute_mass(x):
        pt, eta, e = x[:, :, 0], x[:, :, 1], x[:, :, 3]
        p = pt * np.cosh(eta)
        m = np.sqrt(np.clip(e**2 - p**2, 0, None))
        return m

    def compute(self, x, weight, mask, pt0, m0, mjj):
        x = np.copy(x[:, :, :4])
        m = self.compute_mass(x)
        x[:, :, 0] = x[:, :, 0] * weight
        x[:,:,3] = m
        #print(x[:,:,3])
        #x[:, :, 3] = 0 # temporary override to approximate mass 
        x = x.astype(np.float64)
        n_batch = x.shape[0] 
        for i in range(n_batch):
            evt = x[i][np.logical_and(mask[i].astype(bool), x[i,:,0]>0)]
            evt = np.core.records.fromarrays(
                    evt.T, 
                    names='pt, eta, phi, m',
                    formats='f8, f8, f8, f8'
                )
            seq = pyjet.cluster(evt, R=0.4, p=-1)
            jets = seq.inclusive_jets()
            if len(jets) > 0:
                self.dists['pt'].append(jets[0].pt - pt0[i])
                self.dists_2['pt'][0].append(pt0[i])
                self.dists_2['pt'][1].append(jets[0].pt)

                self.dists['m'].append(jets[0].mass - m0[i])
                self.dists_2['m'][0].append(m0[i])
                self.dists_2['m'][1].append(jets[0].mass)

                if len(jets) > 1:
                    j0, j1 = jets[:2]
                    mjj_pred = np.sqrt(
                            (j0.e + j1.e) ** 2 
                            - (j0.px + j1.px) ** 2
                            - (j0.py + j1.py) ** 2
                            - (j0.pz + j1.pz) ** 2
                        )
                else:
                    mjj_pred = 0 
                if mjj[i] > 0:
                    self.dists['mjj'].append(mjj_pred - mjj[i])
                    self.dists_2['mjj'][0].append(mjj[i])
                    self.dists_2['mjj'][1].append(mjj_pred)

    def plot(self, path):
        for k, data in self.dists.items():
            plt.clf()
            plt.hist(data, bins=self.bins[k])
            plt.xlabel(f'Predicted-True {self.labels[k]} [GeV]')
            for ext in ('pdf', 'png'):
                plt.savefig(f'{path}_{k}_err.{ext}')
            with open(f'{path}_{k}_err.pkl', 'wb') as fpkl:
                pickle.dump(
                        {'data': data, 'bins':self.bins[k]},
                        fpkl
                    )

        for k, data in self.dists_2.items():
            plt.clf()
            plt.hist2d(data[0], data[1], bins=self.bins_2[k])
            plt.xlabel(f'True {self.labels[k]} [GeV]')
            plt.ylabel(f'Predicted {self.labels[k]} [GeV]')
            for ext in ('pdf', 'png'):
                plt.savefig(f'{path}_{k}_corr.{ext}')



class METResolution(object):
    def __init__(self, bins=np.linspace(-100, 100, 40)):
        self.bins = bins
        self.bins_2 = (0, 400)
        self.bins_met1 = (0, 400)
        self.reset()

    def reset(self):
        self.dist = None
        self.dist_p = None
        self.dist_pup = None
        self.dist_2 = None
        self.dist_met = None
        self.dist_pred = None
        self.dist_2_p = None
        self.dist_2_pup = None

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

    def compute(self, pf, pup, gm, pred, weight=None):
        res = (pred - gm)
        res_p = (pf - gm) 
        res_pup = (pup - gm)

        hist, _ = np.histogram(res, bins=self.bins)
        hist_met, self.bins_met = np.histogram(gm, bins=np.linspace(*(self.bins_met1) + (100,)))
        hist_pred, self.bins_pred = np.histogram(pred, bins=np.linspace(*(self.bins_met1) + (100,)))
        hist_p, _ = np.histogram(res_p, bins=self.bins)
        hist_pup, _ = np.histogram(res_pup, bins=self.bins)
        hist_2, _, _ = np.histogram2d(gm, pred, bins=100, range=(self.bins_2, self.bins_2))
        hist_2_p, _, _ = np.histogram2d(gm, pf, bins=100, range=(self.bins_2, self.bins_2))
        if self.dist is None:
            self.dist = hist + EPS
            self.dist_met = hist_met + EPS
            self.dist_pred = hist_pred + EPS
            self.dist_p = hist_p + EPS
            self.dist_pup = hist_pup + EPS
            self.dist_2 = hist_2 + EPS
            self.dist_2_p = hist_2_p + EPS
        else:
            self.dist += hist
            self.dist_met += hist_met
            self.dist_pred += hist_pred
            self.dist_p += hist_p
            self.dist_pup += hist_pup
            self.dist_2 += hist_2
            self.dist_2_p += hist_2_p

    @staticmethod 
    def _compute_moments(x, dist):
        dist = dist / np.sum(dist)
        mean = np.sum(x * dist) 
        var = np.sum(np.power(x - mean, 2) * dist) 
        return mean, var

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5

        mean, var = self._compute_moments(x, self.dist)
        mean_p, var_p = self._compute_moments(x, self.dist_p)
        mean_pup, var_pup = self._compute_moments(x, self.dist_pup)

        label = r'Model ($\delta=' + f'{mean:.1f}' + r'\pm' + f'{np.sqrt(var):.1f})$'
        plt.hist(x=x, weights=self.dist, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{mean_pup:.1f}' + r'\pm' + f'{np.sqrt(var_pup):.1f})$'
        plt.hist(x=x, weights=self.dist_pup, label=label, histtype='step', bins=self.bins)

        label = r'Ground Truth ($\delta=' + f'{mean_p:.1f}' + r'\pm' + f'{np.sqrt(var_p):.1f})$'
        plt.hist(x=x, weights=self.dist_p, label=label, histtype='step', bins=self.bins, linestyle='--')

        plt.xlabel('Predicted-True MET [GeV]')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        plt.clf()
        x = (self.bins_met[:-1] + self.bins_met[1:]) * 0.5
        plt.hist(x=x, weights=self.dist_met, bins=self.bins_met)
        plt.xlabel('True MET')
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_true.' + ext)

        plt.clf()
        x = (self.bins_pred[:-1] + self.bins_pred[1:]) * 0.5
        plt.hist(x=x, weights=self.dist_pred, bins=self.bins_pred)
        plt.xlabel('Predicted MET')
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_pred.' + ext)

        plt.clf()
        self.dist_2 = np.ma.masked_where(self.dist_2 < 0.5, self.dist_2)
        plt.imshow(self.dist_2.T, vmin=0.5, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.xlabel('True MET [GeV]')
        plt.ylabel('Predicted MET [GeV]')
        plt.colorbar()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_corr.' + ext)
        plt.clf()
        self.dist_2_p = np.ma.masked_where(self.dist_2_p < 0.5, self.dist_2_p)
        plt.imshow(self.dist_2_p.T, vmin=0.5, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.colorbar()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_corr_pf.' + ext)
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
        res =  met - gm # (met / gm) - 1
        return res

    def compute(self, pt, phi, w, y, baseline, gm):
        res = self._compute_res(pt, phi, w, gm)
        res_t = self._compute_res(pt, phi, y, gm)
        res_p = self._compute_res(pt, phi, baseline, gm)

        hist, _ = np.histogram(res, bins=self.bins)
        hist_p, _ = np.histogram(res_p, bins=self.bins)
        hist_met, _ = np.histogram(res_t, bins=self.bins)
        if self.dist is None:
            self.dist = hist + EPS
            self.dist_p = hist_p + EPS
            self.dist_met = hist_met + EPS
        else:
            self.dist += hist
            self.dist_p += hist_p
            self.dist_met += hist_met

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5

        mean, var = self._compute_moments(x, self.dist)
        mean_p, var_p = self._compute_moments(x, self.dist_p)
        mean_met, var_met = self._compute_moments(x, self.dist_met)

        label = r'Model ($\delta=' + f'{mean:.1f}' + r'\pm' + f'{np.sqrt(var):.1f})$'
        plt.hist(x=x, weights=self.dist, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{mean_p:.1f}' + r'\pm' + f'{np.sqrt(var_p):.1f})$'
        plt.hist(x=x, weights=self.dist_p, label=label, histtype='step', bins=self.bins)

        label = r'Truth+PF ($\delta=' + f'{mean_met:.1f}' + r'\pm' + f'{np.sqrt(var_met):.1f})$'
        plt.hist(x=x, weights=self.dist_met, label=label, histtype='step', bins=self.bins)

        plt.xlabel('(Predicted-True)')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        return {'model': (mean, np.sqrt(var)), 'puppi': (mean_p, np.sqrt(var_p))}


class PapuMetrics(object):
    def __init__(self):
        self.loss_calc = nn.MSELoss(
                reduction='none'
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
        self.hists = {}
        self.bins = {}

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

    def add_values(self, val, key, w=None, lo=0, hi=1):
        hist, bins = np.histogram(
                val, bins=np.linspace(lo, hi, 100), weights=w)
        if key not in self.hists:
            self.hists[key] = hist + EPS
            self.bins[key] = bins
        else:
            self.hists[key] += hist

    def compute(self, yhat, y, w=None, m=None):
        yhat = yhat.view(-1)
        y = y.view(-1)
        loss = self.loss_calc(yhat, y)
        if w is not None:
            wv = w.view(-1)
            loss *= wv
        if m is None:
            m = torch.ones_like(y, dtype=bool)
        m = m.view(-1)
        loss *= m

        loss = torch.mean(loss)
        self.loss += t2n(loss).mean()

        m = t2n(m).astype(bool)
        y = t2n(y)[m]
        if w is not None:
            w = t2n(w).reshape(-1)[m]
        yhat = t2n(yhat)[m]
        n_particles = m.sum()

        # let's define positive/negative by >/< 0.5
        y_bin = y > 0.5 
        yhat_bin = yhat > 0.5

        acc = (y_bin == yhat_bin).sum() / n_particles
        self.acc += acc

        n_pos = y_bin.sum()
        pos_acc = (y_bin == yhat_bin)[y_bin].sum() / n_pos 
        self.pos_acc += pos_acc
        n_neg = (~y_bin).sum()
        neg_acc = (y_bin == yhat_bin)[~y_bin].sum() / n_neg 
        self.neg_acc += neg_acc

        self.n_pos += n_pos
        self.n_particles += n_particles

        self.n_steps += 1

        self.add_values(
            y, 'truth', w, -0.2, 1.2) 
        self.add_values(
            yhat, 'pred', w, -0.2, 1.2) 
        self.add_values(
            yhat-y, 'err', w, -2, 2)

        return loss, acc

    def mean(self):
        return ([x / self.n_steps 
                 for x in [self.loss, self.acc, self.pos_acc, self.neg_acc]]
                + [self.n_pos / self.n_particles])

    def plot(self, path):
        plt.clf()
        bins = self.bins['truth'] 
        x = (bins[:-1] + bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,
                'bins': bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists['truth'], label='Truth', **hist_args)
        plt.hist(weights=self.hists['pred'], label='Pred', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel(r'$E_{\mathrm{hard}}/E_{\mathrm{tot.}}$')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_e.' + ext)

        plt.clf()
        bins = self.bins['err'] 
        x = (bins[:-1] + bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,
                'bins': bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists['err'], label='Error', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel(r'Prediction - Truth')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_err.' + ext)

