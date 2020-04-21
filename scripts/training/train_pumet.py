#!/usr/bin/env python3
from grapple import utils

p = utils.ArgumentParser()
p.add_args(
    '--dataset_pattern', '--output', ('--n_epochs', p.INT),
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--label_size', p.INT), ('--num_hidden_layers', p.INT), ('--batch_size', p.INT),
    ('--mask_charged', p.STORE_TRUE), ('--lr', {'type': float}),
    ('--lr_schedule', p.STORE_TRUE), '--plot',
    ('--pt_weight', p.STORE_TRUE), ('--num_max_files', p.INT),
    ('--num_max_particles', p.INT), ('--dr_adj', p.FLOAT),
    ('--met_poly_degree', p.INT), ('--met_layers', p.INT),
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
    snapshot = utils.Snapshot(config.output, config)
    logger.info(f'Saving output to {snapshot.path}')

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

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr = torch.optim.lr_scheduler.ExponentialLR(opt, config.lr_decay)
    # lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         opt, 
    #         factor=config.lr_decay,
    #         patience=3
    #     )
    metrics = METMetrics(device)
    metrics_puppi = METMetrics(device, softmax=False)
    met = METResolution()

    # model, opt = amp.initialize(model, opt, opt_level='O1')

    if not os.path.exists(config.plot):
        os.makedirs(config.plot)

    for e in range(config.n_epochs):
        logger.info(f'Epoch {e}: Start')
        current_lr = [group['lr'] for group in opt.param_groups][0]
        logger.info(f'Epoch {e}: Current LR = {current_lr}')
        logger.info(f'Epoch {e}: N_particles = {ds.n_particles}')

        model.unfreeze_all()
        if e < 5:
            # only fit PU
            model.freeze_met()
            metrics.met_loss_weight = metrics_puppi.met_loss_weight = 0.
        elif e < 10:
            # only fit MET, based on a frozen encoder
            model.freeze_pu()
            metrics.met_loss_weight = metrics_puppi.met_loss_weight = 1

        model.train()
        metrics.reset()
        metrics_puppi.reset()
        met.reset()
        avg_loss_tensor = 0
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

            opt.zero_grad()
            yhat, methat = model(x, q=q, y=y_float, mask=m, particle_mask=particle_mask)
            if config.pt_weight:
                weight = x[:, :, 0] / x[:, 0, 0].reshape(-1, 1)
                weight = weight ** 2
            else:
                weight = None
            # weight = (y == 1) + 1
            loss, _ = metrics.compute(yhat, y, orig_y, gm, methat, w=weight, m=neutral_mask)
            loss.backward()

            # p = torch.ones_like(y, dtype=np.float) # override puppi - broken in data
            #                        # this reduces to uncorrected PF
            p = torch.stack([1-p, p], dim=-1)
            metrics_puppi.compute(p, y, orig_y, gm, puppimet, w=weight, m=neutral_mask)

            # with amp.scale_loss(loss, opt) as scaled_loss:
            #     scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            avg_loss_tensor += loss

            methat = t2n(methat)
            gm = t2n(gm)
            puppimet = t2n(puppimet)
            met.compute(puppimet, gm, methat)

        avg_loss_tensor /= n_batch
        lr.step()

        plot_path = f'{config.plot}/resolution_{e:03d}'

        metrics.plot(plot_path + '_model')
        metrics_puppi.plot(plot_path + '_puppi')
        met.plot(plot_path + '_met')

        avg_loss, avg_acc, avg_posacc, avg_negacc, avg_posfrac = metrics.mean()
        logger.info(f'Epoch {e}: Average fraction of hard particles = {avg_posfrac}')
        logger.info(f'Epoch {e}: MODEL:')
        logger.info(f'Epoch {e}: Loss = {avg_loss}; Accuracy = {avg_acc}')
        logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')
        
        avg_loss, avg_acc, avg_posacc, avg_negacc, _ = metrics_puppi.mean()
        logger.info(f'Epoch {e}: PUPPI:')
        logger.info(f'Epoch {e}: Loss = {avg_loss}; Accuracy = {avg_acc}')
        logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')

        torch.save(model.state_dict(), snapshot.get_path(f'model_weights_epoch{e}.pt'))

        # ds.n_particles = min(2000, ds.n_particles + 50)
