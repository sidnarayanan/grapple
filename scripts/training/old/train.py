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
)
config = p.parse_args()

t2n = utils.t2n

from grapple.data import PUDataset, DataLoader
from grapple.metrics import * 
from grapple.model import Joe, Bruno

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

    logger.info(f'Reading dataset at {config.dataset_pattern}')
    ds = PUDataset(config)
    dl = DataLoader(ds, batch_size=config.batch_size, collate_fn=PUDataset.collate_fn)

    logger.info(f'Building model')
    model = Bruno(config)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr = torch.optim.lr_scheduler.ExponentialLR(opt, config.lr_decay)
    # lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         opt, 
    #         factor=config.lr_decay,
    #         patience=3
    #     )
    metrics = Metrics(device)
    metrics_puppi = Metrics(device, softmax=False)
    met = ParticleMETResolution()

    # model, opt = amp.initialize(model, opt, opt_level='O1')

    if not os.path.exists(config.plot):
        os.makedirs(config.plot)

    for e in range(config.n_epochs):
        logger.info(f'Epoch {e}: Start')
        current_lr = [group['lr'] for group in opt.param_groups][0]
        logger.info(f'Epoch {e}: Current LR = {current_lr}')
        logger.info(f'Epoch {e}: N_particles = {ds.n_particles}')

        model.train()
        metrics.reset()
        metrics_puppi.reset()
        met.reset()
        avg_loss_tensor = 0
        for n_batch, batch in enumerate(tqdm(dl, total=len(ds) // config.batch_size)):
            if n_batch % 1000 == 0:
                logger.debug(n_batch)
            x = torch.Tensor(batch[0]).to(device)
            y = torch.LongTensor(batch[1]).to(device)
            m = torch.LongTensor(batch[2]).to(device)
            p = torch.Tensor(batch[4]).to(device)
            pfmet = batch[6]
            orig_y = batch[7]
            neutral_mask = (batch[0][:, :, 5] == 0) # neutral 
            gm = batch[5]

            opt.zero_grad()
            yhat = model(x, mask=m)
            if config.pt_weight:
                # weight = np.arange(x.shape[1]) + 1
                # weight = 1. / weight 
                # weight = np.tile(weight, x.shape[0])
                weight = x[:, :, 0] / x[:, 0, 0].reshape(-1, 1)
                weight = weight ** 2
                # weight = torch.Tensor(weight).to(device)
            else:
                weight = None
            # weight = x[:,:,0] if config.pt_weight else None
            # weight = (y == 1) + 1
            loss, _ = metrics.compute(yhat, y, orig_y, w=weight, m=neutral_mask)
            loss.backward()

            # p = torch.ones_like(y, dtype=np.float) # override puppi - broken in data
            #                        # this reduces to uncorrected PF
            p = torch.stack([1-p, p], dim=-1)
            metrics_puppi.compute(p, y, orig_y, w=weight, m=neutral_mask)

            # with amp.scale_loss(loss, opt) as scaled_loss:
            #     scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            avg_loss_tensor += loss

            x = batch[0]
            y = batch[1]
            q = x[:,:,5]
            score = t2n(nn.functional.softmax(yhat, dim=-1)[:,:,1])
            score = utils.rescore(score, y, q, rescale=False)
            met.compute(x[:,:,0], x[:,:,2], score, y, pfmet, gm)

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

        ds.n_particles = min(2000, ds.n_particles + 50)
