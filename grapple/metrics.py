import torch
from torch import nn 
from .utils import t2n 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd 
import seaborn as sns

import pyjet
from pyjet.testdata import get_event

from loguru import logger

EPS = 1e-4

from ._old import *


def square(x):
    return torch.pow(x, 2)


class JetResolution(object):
    bins = {
            'pt': np.linspace(-100, 100, 100),
            'm': np.linspace(-50, 50, 100),
            'mjj': np.linspace(-1000, 1000, 100),
        }
    bins_2 = {
            'pt': np.linspace(0, 500, 100),
            'm': np.linspace(0, 50, 100),
            'mjj': np.linspace(0, 3000, 100),
        }
    labels = {
            'pt': 'Jet $p_T$',
            'm': 'Jet $m$',
            'mjj': '$m_{jj}$',
        }
    methods = {
            'model': r'$\mathrm{PUMA}$',
            'puppi': r'$\mathrm{PUPPI}$',
            'truth': 'Truth+PF',
        }
    def __init__(self):
        self.reset()

    def reset(self):
        self.dists = {k:{m:[] for m in self.methods} for k in ['pt', 'm', 'mjj']}
        self.dists_2 = {k:{m:([], []) for m in self.methods} for k in ['pt', 'm', 'mjj']}
        self.truth_pt = {m:[] for m in self.methods}

    @staticmethod
    def compute_p4(x):
        pt, eta, phi, e = (x[:,:,i] for i in range(4))
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta) 
        p4 = np.stack([e, px, py, pz], axis=-1)
        return p4

    @staticmethod 
    def compute_mass(x):
        pt, eta, e = x[:, :, 0], x[:, :, 1], x[:, :, 3]
        p = pt * np.cosh(eta)
        m = np.sqrt(np.clip(e**2 - p**2, 0, None))
        return m

    def compute(self, x, weights, mask, pt0, m0=None, mjj=None):
        for k,v in weights.items():
            self._internal_compute(k, x, v, mask, pt0, m0, mjj)

    def _internal_compute(self, tag, x, weight, mask, pt0, m0=None, mjj=None):
        p4 = self.compute_p4(x[:, :, :4])
        p4 *= weight[:, :, None]
        p4 = p4.astype(np.float64)
        n_batch = p4.shape[0] 
        pt = x[:, :, 0] * weight
        for i in range(n_batch):
            particle_mask = pt[i, :] > 0
            particle_mask = np.logical_and(
                    particle_mask,
                    np.logical_and(
                        ~np.isnan(p4[i]).sum(-1),
                        ~np.isinf(p4[i]).sum(-1)
                    )
                )
            evt = p4[i][np.logical_and(
                                mask[i].astype(bool),
                                particle_mask
                            )]
            evt = np.core.records.fromarrays(
                    evt.T, 
                    names='E, px, py, pz',
                    formats='f8, f8, f8, f8'
                )
            seq = pyjet.cluster(evt, R=0.4, p=-1, ep=True)
            jets = seq.inclusive_jets()
            if len(jets) > 0:
                self.dists['pt'][tag].append(jets[0].pt - pt0[i])
                self.truth_pt[tag].append(pt0[i])
                self.dists_2['pt'][tag][0].append(pt0[i])
                self.dists_2['pt'][tag][1].append(jets[0].pt)

                if m0 is not None:
                    self.dists['m'].append(jets[0].mass - m0[i])
                    self.dists_2['m'][tag][0].append(m0[i])
                    self.dists_2['m'][tag][1].append(jets[0].mass)

                if mjj is not None:
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
                        self.dists['mjj'][tag].append(mjj_pred - mjj[i])
                        self.dists_2['mjj'][tag][0].append(mjj[i])
                        self.dists_2['mjj'][tag][1].append(mjj_pred)

    @staticmethod 
    def _compute_moments(x):
        return np.mean(x), np.std(x)

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5

        mean_p, var_p = self._compute_moments(x, self.dist_p)
        mean_pup, var_pup = self._compute_moments(x, self.dist_pup)


    def plot(self, path):
        for k, m_data in self.dists.items():
            plt.clf()
            for m, data in m_data.items():
                if len(data) == 0:
                    continue
                mean, var = self._compute_moments(data)
                label = fr'{self.methods[m]} ($\delta=' + f'{mean:.1f}' + r'\pm' + f'{np.sqrt(var):.1f})$'
                plt.hist(data, bins=self.bins[k], label=label,
                         histtype='step')
            plt.legend()
            plt.xlabel(f'Predicted-True {self.labels[k]} [GeV]')
            for ext in ('pdf', 'png'):
                plt.savefig(f'{path}_{k}_err.{ext}')

        plt.clf()
        dfs = []
        lo, hi = 0, 500
        n_bins = 10
        bins = np.linspace(lo, hi, n_bins) 
        for m,m_label in self.methods.items():
            truth = np.digitize(self.truth_pt[m], bins)
            truth = (truth * (hi - lo) / n_bins) + lo
            df = pd.DataFrame({
                'x': truth, 
                'y': self.dists['pt'][m],
                'Method': [m_label] * truth.shape[0]
            })
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        sns.boxplot(
                x='x', y='y', hue='Method', data=df,
                order=(bins * (hi - lo) / n_bins) + lo,
            )
        plt.xlabel(rf'True {self.labels["pt"]} [GeV]')
        plt.ylabel(rf'Error {self.labels["pt"]} [GeV]')
        for ext in ('pdf', 'png'):
            plt.savefig(f'{path}_differr.{ext}')

        for k, m_data in self.dists_2.items():
            for m, data in m_data.items():
                if len(data) == 0:
                    continue
                plt.clf()
                plt.hist2d(data[0], data[1], bins=self.bins_2[k], cmin=0.1)
                plt.xlabel(f'True {self.labels[k]} [GeV]')
                plt.ylabel(f'Predicted {self.labels[k]} [GeV]')
                for ext in ('pdf', 'png'):
                    plt.savefig(f'{path}_{k}_{m}_corr.{ext}')



class METResolution(object):
    methods = {
            'model': r'$\mathrm{PUMA}$',
            'puppi': r'$\mathrm{PUPPI}$',
            'truth': 'Truth+PF',
        }
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

        label = r'$\mathrm{PUMA}$ ($\delta=' + f'{mean:.1f}' + r'\pm' + f'{np.sqrt(var):.1f})$'
        plt.hist(x=x, weights=self.dist, label=label, histtype='step', bins=self.bins)

        label = r'$\mathrm{PUPPI}$ ($\delta=' + f'{mean_pup:.1f}' + r'\pm' + f'{np.sqrt(var_pup):.1f})$'
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
    def __init__(self, which='mag', bins=np.linspace(-100, 100, 40)):
        super().__init__(bins)
        self.bins_phi = np.linspace(0, 3.142, 40)
        self.which = which 

    def reset(self):
        self.dist = None
        self.dist_p = None
        self.dist_pup = None
        self.dist_2 = None
        self.dist_met = None
        self.dist_pred = None
        self.dist_2_p = None
        self.dist_2_pup = None

        self.pred = {k: [] for k in self.methods}
        self.truth = [] 
        self.predphi = {k: [] for k in self.methods}
        self.truthphi = [] 

    @staticmethod
    def _compute_res(pt, phi, w, gm, gmphi, which):
        pt = pt * w
        if which in ('mag', 'x'):
            px = pt * np.cos(phi)
            metx = -np.sum(px, axis=-1)
        if which in ('mag', 'y'):
            py = pt * np.sin(phi)
            mety = -np.sum(py, axis=-1)
        if which == 'mag':
            met = np.sqrt(np.power(metx, 2) + np.power(mety, 2))
            met_vec = np.stack([metx, mety], axis=-1)
            gm_vec = np.stack([gm*np.cos(gmphi), gm*np.sin(gmphi)], axis=-1)
            resphi = np.arccos(
                    np.einsum('ij,ij->i', met_vec, gm_vec) / (met * gm)
                )
        elif which == 'x':
            met = metx 
            resphi = np.zeros_like(met)
        elif which == 'y':
            met = mety
            resphi = np.zeros_like(met)
        res =  met - gm # (met / gm) - 1
        return res, resphi

    def compute(self, pt, phi, w, y, baseline, gm, gmphi):
        res, resphi = self._compute_res(pt, phi, w, gm, gmphi, self.which)
        res_t, resphi_t = self._compute_res(pt, phi, y, gm, gmphi, self.which)
        res_p, resphi_p = self._compute_res(pt, phi, baseline, gm, gmphi, self.which)

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

        self.pred['model'] += res.tolist() 
        self.pred['puppi'] += res_p.tolist() 
        self.pred['truth'] += res_t.tolist() 
        self.truth += gm.tolist()
        self.predphi['model'] += resphi.tolist() 
        self.predphi['puppi'] += resphi_p.tolist() 
        self.predphi['truth'] += resphi_t.tolist() 
        self.truthphi += gmphi.tolist()

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5

        mean, var = self._compute_moments(x, self.dist)
        mean_p, var_p = self._compute_moments(x, self.dist_p)
        mean_met, var_met = self._compute_moments(x, self.dist_met)

        label = r'$\mathrm{PUMA}$ ($\delta=' + f'{mean:.1f}' + r'\pm' + f'{np.sqrt(var):.1f})$'
        plt.hist(x=x, weights=self.dist, label=label, histtype='step', bins=self.bins)

        label = r'$\mathrm{PUPPI}$ ($\delta=' + f'{mean_p:.1f}' + r'\pm' + f'{np.sqrt(var_p):.1f})$'
        plt.hist(x=x, weights=self.dist_p, label=label, histtype='step', bins=self.bins)

        label = r'Truth+PF ($\delta=' + f'{mean_met:.1f}' + r'\pm' + f'{np.sqrt(var_met):.1f})$'
        plt.hist(x=x, weights=self.dist_met, label=label, histtype='step', bins=self.bins)

        plt.xlabel(r'Predicted-True $p_\mathrm{T}^\mathrm{miss}$')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        plt.clf()
        dfs = []
        lo, hi = 0, 500
        n_bins = 10
        bins = np.linspace(lo, hi, n_bins) 
        truth = np.digitize(self.truth, bins)
        truth = (truth * (hi - lo) / n_bins) + lo
        for m,m_label  in self.methods.items():
            df = pd.DataFrame({
                'x': truth, 
                'y': self.pred[m],
                'Method': [m_label] * truth.shape[0]
            })
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        sns.boxplot(
                x='x', y='y', hue='Method', data=df,
                order=(bins * (hi - lo) / n_bins) + lo,
            )
        plt.xlabel(r'True $p_\mathrm{T}^\mathrm{miss}$ [GeV]')
        plt.ylabel(r'Error $p_\mathrm{T}^\mathrm{miss}$ [GeV]')
        for ext in ('pdf', 'png'):
            plt.savefig(f'{path}_differr.{ext}')
        
        plt.clf()
        dfs = []
        for m,m_label  in self.methods.items():
            df = pd.DataFrame({
                'x': truth, 
                'y': self.predphi[m],
                'Method': [m_label] * truth.shape[0]
            })
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        sns.boxplot(
                x='x', y='y', hue='Method', data=df,
                order=(bins * (hi - lo) / n_bins) + lo,
            )
        plt.xlabel(r'True $p_\mathrm{T}^\mathrm{miss}$ [GeV]')
        plt.ylabel(r'Error $\phi^\mathrm{miss}$ [GeV]')
        for ext in ('pdf', 'png'):
            plt.savefig(f'{path}_diffphierr.{ext}')
        

        return {'model': (mean, np.sqrt(var)), 'puppi': (mean_p, np.sqrt(var_p))}


class PapuMetrics(object):
    def __init__(self, beta=False, met_weight=0):
        self.met_weight = met_weight
        self.beta = beta 
        if not self.beta:
            self.loss_calc = nn.MSELoss(
                    reduction='none'
                )
        else:
            def neglogbeta(p, q, y):
                loss = torch.lgamma(p + q)
                loss -= torch.lgamma(p) + torch.lgamma(q)
                loss += (p - 1) * torch.log(y + EPS)
                loss += (q - 1) * torch.log(1 - y + EPS)
                return -loss 
            self.loss_calc = neglogbeta
            def beta_mean(p, q):
                return p / (p + q)
            self.beta_mean = beta_mean 
            def beta_std(p, q):
                return torch.sqrt(p*q / ((p+q)**2 * (p+q+1)))
            self.beta_std = beta_std
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

    def _compute_met_constraint(self, yhat, y, x):
        if self.met_weight == 0:
            return 0 

        assert (x is not None)

        pt, phi = x[:,:,0], x[:,:,2]
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)

        def calc(scale, p):
            print(scale.shape, pt.shape, p.shape)
            return torch.sum(scale * p, dim=-1)

        yhat = yhat.reshape(pt.shape)
        y = y.reshape(pt.shape)

        pred_metx = calc(yhat, px)
        pred_mety = calc(yhat, py)
        true_metx = calc(y, px)
        true_mety = calc(y, py)

        err = square(true_metx - pred_metx) + square(true_mety - pred_mety)
        err = err.mean()
        return self.met_weight * err


    def compute(self, yhat, y, w=None, m=None, plot_m=None, x=None):
        y = y.view(-1)
        if not self.beta:
            yhat = yhat.view(-1)
            loss = self.loss_calc(yhat, y)
        else:
            yhat = yhat + EPS
            p, q = yhat[:, :, 0], yhat[:, :, 1]
            p, q = p.view(-1), q.view(-1)
            loss = self.loss_calc(p, q, y)
            yhat = self.beta_mean(p, q)
            yhat_std = self.beta_std(p, q)
        yhat = torch.clamp(yhat, 0 , 1)
        if w is not None:
            wv = w.view(-1)
            loss *= wv
        if m is None:
            m = torch.ones_like(y, dtype=bool)
        if plot_m is None:
            plot_m = m
        m = m.view(-1).float()
        plot_m = m.view(-1)
        loss *= m

        nan_mask = t2n(torch.isnan(loss)).astype(bool)

        loss = torch.mean(loss)
        loss += self._compute_met_constraint(yhat, y, x)

        self.loss += t2n(loss).mean()

        if nan_mask.sum() > 0:
            yhat = t2n(yhat)
            logger.info(nan_mask)
            logger.info(yhat[nan_mask])
            if self.beta:
                p, q = t2n(p), t2n(q)
                logger.info(p[nan_mask])
                logger.info(q[nan_mask])

        plot_m = t2n(plot_m).astype(bool)
        y = t2n(y)[plot_m]
        if w is not None:
            w = t2n(w).reshape(-1)[plot_m]
        yhat = t2n(yhat)[plot_m]
        n_particles = plot_m.sum()

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

