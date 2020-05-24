#!/usr/bin/env python3
from grapple import utils

p = utils.ArgumentParser()
p.add_args(
    ('--dataset_pattern', p.MANY), '--weights', ('--mask_charged', p.STORE_TRUE), 
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--label_size', p.INT), ('--num_hidden_layers', p.INT), ('--batch_size', p.INT),
    ('--met_poly_degree', p.INT), 
    ('--met_layers', p.INT),
     '--plot',
    ('--pt_weight', p.STORE_TRUE), 
    ('--num_max_files', p.INT),
    ('--num_max_particles', p.INT), 
    ('--dr_adj', p.FLOAT),
)
config = p.parse_args()

t2n = utils.t2n

from grapple.data import PUDataset, DataLoader
from grapple.metrics import * 
from grapple.model import Joe, Bruno, Agnes

from apex import amp
from tqdm import tqdm, trange
from loguru import logger
import torch
from torch import nn 
from torch.utils.data import RandomSampler
import os


if __name__ == '__main__':
    # snapshot = utils.Snapshot(config.output, config)
    # logger.info(f'Saving output to {snapshot.path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    to_t = lambda x: torch.Tensor(x).to(device)
    to_lt = lambda x: torch.LongTensor(x).to(device)

    logger.info(f'Reading dataset at {config.dataset_pattern}')
    ds = PUDataset(config)
    dl = DataLoader(ds, batch_size=config.batch_size, collate_fn=PUDataset.collate_fn)

    logger.info(f'Building model')
    model = Agnes(config)
    model = model.to(device)

    model.load_state_dict(torch.load(config.weights)) 

    metrics = METMetrics(device)
    metrics_puppi = METMetrics(device, softmax=False)
    met = METResolution()
    jet = JetResolution()
    jet_puppi = JetResolution()
    jet_pf = JetResolution()

    if not os.path.exists(config.plot):
        os.makedirs(config.plot)

    model.eval()
    metrics.reset()
    metrics_puppi.reset()
    met.reset()

    for n_batch, batch in enumerate(tqdm(dl, total=len(ds) // config.batch_size)):
        x = to_t(batch['x'])
        y = to_lt(batch['y'])
        y_float = to_t(batch['x'][:,:,4]==0)
        m = to_lt(batch['adj_mask'])
        p = to_t(batch['p'])
        puppimet = to_t(batch['puppimet'])
        q = to_lt(batch['q'])
        gm = to_t(batch['genmet'])
        particle_mask = to_lt(batch['mask'])

        orig_y = batch['orig_y']
        neutral_mask = (batch['x'][:,:,5] == 0)

        yhat, methat, pred_weights = model(
                x, q=q, y=y_float, mask=m, particle_mask=particle_mask, return_weights=True
            )
        if config.pt_weight:
            weight = x[:, :, 0] / x[:, 0, 0].reshape(-1, 1)
            weight = weight ** 2
        else:
            weight = None
        metrics.compute(yhat, y, orig_y, gm, methat, w=weight, m=neutral_mask)

        p = torch.stack([1-p, p], dim=-1)
        metrics_puppi.compute(p, y, orig_y, gm, puppimet, w=weight, m=neutral_mask)

        methat = t2n(methat)
        gm = t2n(gm)
        met.compute(batch['pfmet'], batch['puppimet'], gm, methat)

        # pred_weights = torch.nn.functional.softmax(yhat, dim=-1)[:, :, 1]

        pred_weights = t2n(pred_weights)
        # pred_weights = (pred_weights > 0.5).astype(float)

        pred_weights = utils.rescore(pred_weights, batch['q'], t2n(y_float), rescale=False)

        jet.compute(batch['x'], pred_weights, batch['mask'], batch['jpt0'], batch['jm0'], batch['mjj']) 
        jet_puppi.compute(batch['x'], batch['p'], batch['mask'], batch['jpt0'], batch['jm0'], batch['mjj']) 
        jet_pf.compute(batch['x'], np.ones_like(batch['p']), batch['mask'], batch['jpt0'], batch['jm0'], batch['mjj']) 


    plot_path = f'{config.plot}/inference'

    metrics.plot(plot_path + '_model')
    metrics_puppi.plot(plot_path + '_puppi')
    met.plot(plot_path + '_met')
    jet.plot(plot_path + '_jet_model')
    jet_puppi.plot(plot_path + '_jet_puppi')
    jet_pf.plot(plot_path + '_jet_pf')

    avg_loss, avg_acc, avg_posacc, avg_negacc, avg_posfrac = metrics.mean()
    logger.info(f'Inference: Average fraction of hard particles = {avg_posfrac}')
    logger.info(f'Inference: MODEL:')
    logger.info(f'Inference: Accuracy = {avg_acc}')
    logger.info(f'Inference: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')
    
    avg_loss, avg_acc, avg_posacc, avg_negacc, _ = metrics_puppi.mean()
    logger.info(f'Inference: PUPPI:')
    logger.info(f'Inference: Accuracy = {avg_acc}')
    logger.info(f'Inference: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')
