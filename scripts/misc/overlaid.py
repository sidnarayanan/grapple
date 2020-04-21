#!/usr/bin/env python

from os import getenv
import os

from PandaCore.Utils.root import root 
from PandaCore.Tools.Misc import *
from PandaCore.Utils.load import *
import PandaAnalysis.T3.job_utilities as utils
from PandaAnalysis.Flat.analysis import * 
from PandaCore.Tools.root_interface import *
from PandaCore.Tools.script import * 
import numpy as np
from time import sleep
from glob import glob

Load('PandaAnalysisFlat')
data_dir = getenv('CMSSW_BASE') + '/src/PandaAnalysis/data/'
out_dir = getenv('SUBMIT_OUTDIR')

def post_fn(f_aux, out_dir):
    logger.info(f_aux)

    f_npz = out_dir + '/' + f_aux.split('/')[-1].replace('.root', '.npz')
    arr = read_files([f_aux], 
                     branches=['kinematics', 'npv', 'genMet','genMetPup', 'genMetCalc',
                               'genMetPhi', 'genJetPt0', 'genJetM0', 'genMjj'],
                     treename='events')
    k = arr['kinematics']
    k = np.stack(k, axis=0)
    shape = k.shape
    k = np.stack(k.flatten(), axis=0)
    k = k.reshape(shape + (k.shape[-1],))

    p = k[:,:,6]
    q = k[:,:,5]
    y = k[:,:,4] == 0

    x = k[:,:,:6]
    x[:,:,4][(q == 0)] = -1 # if it's a neutral particle, mask the vertex ID

    met = arr['genMet']
    metphi = arr['genMetPhi']
    jetpt0 = arr['genJetPt0']
    jetm0 = arr['genJetM0']
    mjj = arr['genMjj']
    npv = arr['npv']
    puppimet = arr['genMetPup']
    pfmet = arr['genMetCalc']

    np.savez(f_npz, x=x, y=y, q=q, p=p, 
                    met=met, metphi=metphi, 
                    jpt0=jetpt0, jm0=jetm0, mjj=mjj,
                    npv=npv, puppimet=puppimet, pfmet=pfmet)

if __name__ == "__main__":
    args = parse(('--infiles', MANY), '--outdir') 

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    for f in args.infiles:
        post_fn(f, args.outdir)
