#!/usr/bin/env python3
from grapple import utils

p = utils.ArgumentParser()
p.add_args(
    '--dataset_pattern', '--output', ('--n_epochs', p.INT),
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


def scale_fn(c, decay):
    return decay ** c


if __name__ == '__main__':

    snapshot = utils.Snapshot(config.output, config)
    logger.info(f'Saving output to {snapshot.path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    to_t = lambda x: torch.Tensor(x).to(device)
    to_lt = lambda x: torch.LongTensor(x).to(device)

    if config.grad_acc is not None:
        config.batch_size //= config.grad_acc
    else:
        config.grad_acc = 1

    if torch.cuda.device_count() > 1:
        config.batch_size *= torch.cuda.device_count()

    logger.info(f'Reading dataset at {config.dataset_pattern}')
    ds = PapuDataset(config)
    dl = DataLoader(ds, batch_size=config.batch_size, 
                    collate_fn=PapuDataset.collate_fn)
    steps_per_epoch = len(ds) // config.batch_size

    logger.info(f'Building model')

    model = Bruno(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.from_snapshot is not None:
        # original saved file with DataParallel
        state_dicts = torch.load(config.from_snapshot)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        model_state_dict = OrderedDict()
        for k, v in state_dicts['model'].items():
            name = k
            if k.startswith('module'):
                name = k[7:] # remove `module.`
            model_state_dict[name] = v
        # load params
        model.load_state_dict(model_state_dict)

        opt.load_state_dict(state_dict['opt'])

        logger.info(f'Snapshot {config.from_snapshot} loaded.')

    # lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         opt, 
    #         factor=config.lr_decay,
    #         patience=3
    #     )
    metrics = PapuMetrics(config.beta)
    metrics_puppi = PapuMetrics()
    metres = ParticleMETResolution()

    model = model.to(device)
    model, opt = amp.initialize(model, opt, opt_level='O1')
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
            # logger.info(' '.join([str(x) for x in [p.max(), p.min(), q.max(), q.min()]]))
            yhat = p / (p + q + 1e-5)

        score = t2n(torch.clamp(yhat.squeeze(-1), 0, 1))
        charged_mask = ~batch['neutral_mask']
        score[charged_mask] = batch['y'][charged_mask]

        metres.compute(pt=batch['x'][:, :, 0],
                       phi=batch['x'][:, :, 2],
                       w=score,
                       y=batch['y'],
                       baseline=batch['puppi'],
                       gm=genmet)

    avg_loss_tensor /= n_batch

    plot_path = f'{config.plot}/resolution_{e:03d}'

    metrics.plot(plot_path + '_model')
    metrics_puppi.plot(plot_path + '_puppi')
    resolution = metres.plot(plot_path + '_met')

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
