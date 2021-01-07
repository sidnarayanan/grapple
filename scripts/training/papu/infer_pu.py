#!/usr/bin/env python3
from grapple import utils

p = utils.ArgumentParser()
p.add_args(
    '--dataset_pattern', '--output', ('--n_epochs', p.INT),
    '--checkpoint_path',
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--label_size', p.INT), ('--num_hidden_layers', p.INT), ('--batch_size', p.INT),
    ('--mask_charged', p.STORE_TRUE), ('--lr', {'type': float}),
    ('--attention_band', p.INT),
    ('--epoch_offset', p.INT),
    ('--from_snapshot'),
    ('--lr_schedule', p.STORE_TRUE), '--plot',
    ('--pt_weight', p.STORE_TRUE), ('--num_max_files', p.INT),
    ('--num_max_particles', p.INT), ('--dr_adj', p.FLOAT),
    ('--beta', p.STORE_TRUE),
    ('--lr_policy'), ('--grad_acc', p.INT),
)
config = p.parse_args()

t2n = utils.t2n

from grapple.data import PapuDataset, DataLoader
from grapple.metrics import * 
from grapple.model import Joe, Bruno, Agnes, sparse

from tqdm import tqdm, trange
from loguru import logger
import torch
from torch import nn 
from torch.utils.data import RandomSampler
import os
from apex import amp
from functools import partial
from glob import glob
import re


def scale_fn(c, decay):
    return decay ** c


if __name__ == '__main__':

    snapshot = utils.Snapshot(config.output, config)
    logger.info(f'Saving output to {snapshot.path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    to_t = lambda x: torch.Tensor(x).to(device)
    to_lt = lambda x: torch.LongTensor(x).to(device)

    # override
    config.batch_size = 128 

    if torch.cuda.device_count() > 1:
        config.batch_size *= torch.cuda.device_count()

    logger.info(f'Reading dataset at {config.dataset_pattern}')
    ds = PapuDataset(config)
    dl = DataLoader(ds, batch_size=config.batch_size, 
                    collate_fn=PapuDataset.collate_fn)
    steps_per_epoch = len(ds) // config.batch_size

    logger.info(f'Building model')

    def load_checkpoint(path):
        existing_checkpoints = glob(snapshot.get_path(path))
        if existing_checkpoints:
            ckpt = sorted(existing_checkpoints)[-1]
            config.from_snapshot = ckpt
            epoch = int(re.sub(r'.*epoch', '', re.sub(r'\.pt$', '', ckpt)))
            config.epoch_offset = epoch
            
            return True 
        else:
            return False

    if config.from_snapshot is None:
        loaded = load_checkpoint('model_weights_best_epoch*pt')
        if not loaded:
            # if best doesn't exist, take the latest
            loaded = load_checkpoint('model_weights_epoch*pt')

    # if config.from_snapshot is None:
    #     existing_checkpoints = glob(snapshot.get_path('model_weights_epoch*pt'))
    #     if existing_checkpoints:
    #         ckpt = sorted(existing_checkpoints)[-1]
    #         config.from_snapshot = ckpt
    # if config.from_snapshot is not None:
    #     epoch = int(re.sub(r'.*epoch', '', re.sub(r'\.pt$', '', config.from_snapshot)))
    #     config.epoch_offset = epoch

    model = Bruno(config)
    if config.from_snapshot is not None:
        state_dicts = torch.load(config.from_snapshot)
        if 'model' in state_dicts:
            state_dicts = state_dicts['model']
        state_dicts = {re.sub(r'^module\.', '', k):v for k,v in state_dicts.items()}
        model.load_state_dict(state_dicts)

        logger.info(f'Model ckpt {config.from_snapshot} loaded.')

    metrics = PapuMetrics(config.beta)
    metrics_puppi = PapuMetrics()
    metres = ParticleMETResolution()
    metresx = ParticleMETResolution(which='x')
    metresy = ParticleMETResolution(which='y')
    jetres = JetResolution()

    model = model.to(device)
    model = amp.initialize(model, opt_level='O1')
    if torch.cuda.device_count() > 1:
        logger.info(f'Distributing model across {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    if not os.path.exists(config.plot):
        os.makedirs(config.plot)

    e = config.epoch_offset

    logger.info(f'Epoch {e}: Start')
    logger.info(f'Epoch {e}: N_particles = {ds.n_particles}')

    model.eval()
    metrics.reset()
    metrics_puppi.reset()
    metres.reset()
    jetres.reset()

    avg_loss_tensor = 0
    for n_batch, batch in enumerate(tqdm(dl, total=steps_per_epoch)):
        sparse.VERBOSE = (n_batch == 0)

        x = to_t(batch['x'])
        y = to_t(batch['y'])
        m = to_lt(batch['mask'])
        p = to_t(batch['puppi'])
        qm = to_lt(batch['mask'] & batch['neutral_mask'])
        cqm = to_lt(batch['mask'] & ~batch['neutral_mask'])
        genmet = batch['genmet'][:, 0]
        genmetphi = batch['genmet'][:, 1]

        if config.pt_weight:
            weight = x[:, :, 0] / x[:, 0, 0].reshape(-1, 1)
            weight = weight ** 2
        else:
            weight = None

        if True or e < 3:
            loss_mask = m 
        else:
            loss_mask = qm

        with torch.no_grad():
            yhat = model(x, mask=m)
            if not config.beta:
                yhat = torch.sigmoid(yhat)
            else:
                yhat = torch.relu(yhat)
            loss, _ = metrics.compute(yhat, y, w=weight, m=loss_mask, plot_m=qm)

        metrics_puppi.compute(p, y, w=weight, m=loss_mask, plot_m=qm)

        avg_loss_tensor += loss

        if config.beta:
            p, q = yhat[:, :, 0], yhat[:, :, 1]
            yhat = p / (p + q + 1e-5)

        score = t2n(torch.clamp(yhat.squeeze(-1), 0, 1))
        charged_mask = ~batch['neutral_mask']
        score[charged_mask] = batch['y'][charged_mask]

        pt = batch['x'][:, :, 0]
        phi = batch['x'][:, :, 2]

        metres.compute(pt=pt,
                       phi=phi,
                       w=score,
                       y=batch['y'],
                       baseline=batch['puppi'],
                       gm=genmet,
                       gmphi=genmetphi)

        metresx.compute(pt=pt,
                        phi=phi,
                        w=score,
                        y=batch['y'],
                        baseline=batch['puppi'],
                        gm=genmet * np.cos(genmetphi),
                        gmphi=np.zeros_like(genmetphi))

        metresy.compute(pt=pt,
                        phi=phi,
                        w=score,
                        y=batch['y'],
                        baseline=batch['puppi'],
                        gm=genmet * np.sin(genmetphi),
                        gmphi=np.zeros_like(genmetphi))

        jetres.compute(x=batch['x'],
                       weights={
                            'model': score,
                            'puppi': batch['puppi'],
                            'truth': batch['y']
                           },
                       mask=batch['mask'],
                       pt0=batch['jet1'][:,0])


    avg_loss_tensor /= n_batch

    plot_path = f'{config.plot}/resolution_inference'

    metrics.plot(plot_path + '_model')
    metrics_puppi.plot(plot_path + '_puppi')
    resolution = metres.plot(plot_path + '_met')
    metresx.plot(plot_path + '_metx')
    metresy.plot(plot_path + '_mety')
    jetres.plot(plot_path + '_jet')

    avg_loss, avg_acc, avg_posacc, avg_negacc, avg_posfrac = metrics.mean()
    logger.info(f'Epoch {e}: Average fraction of hard particles = {avg_posfrac}')
    logger.info(f'Epoch {e}: MODEL:')
    logger.info(f'Epoch {e}: Loss = {avg_loss}; Accuracy = {avg_acc}')
    logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')
    
    avg_loss, avg_acc, avg_posacc, avg_negacc, _ = metrics_puppi.mean()
    logger.info(f'Epoch {e}: PUPPI:')
    logger.info(f'Epoch {e}: Loss = {avg_loss}; Accuracy = {avg_acc}')
    logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')

    logger.info(f'Epoch {e}: Model MET err = {resolution["model"][0]} +/- {resolution["model"][1]}')
    logger.info(f'Epoch {e}: Puppi MET err = {resolution["puppi"][0]} +/- {resolution["puppi"][1]}')
