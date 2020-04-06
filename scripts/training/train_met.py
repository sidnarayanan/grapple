#!/usr/bin/env python3
from grapple.utils import * 

p = ArgumentParser()
p.add_args(
    '--dataset_pattern', '--output', ('--n_epochs', p.INT),
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--num_hidden_layers', p.INT), ('--batch_size', p.INT),
    ('--mask_charged', p.STORE_TRUE), ('--lr', {'type': float}),
    ('--lr_schedule', p.STORE_TRUE), '--plot',
    ('--min_met', p.FLOAT),
    ('--pt_weight', p.STORE_TRUE)
)
config = p.parse_args()


from grapple.data import PUDataset, DataLoader
from grapple.metrics import * 
from grapple.model import Jane 

from apex import amp
from tqdm import tqdm, trange
from loguru import logger
import torch
from torch import nn 
from torch.utils.data import RandomSampler
from torchsummary import summary
import os


if __name__ == '__main__':
    snapshot = Snapshot(config.output, config)
    logger.info(f'Saving output to {snapshot.path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    logger.info(f'Reading dataset at {config.dataset_pattern}')
    ds = PUDataset(config)
    dl = DataLoader(ds, batch_size=config.batch_size, collate_fn=PUDataset.collate_fn)

    logger.info(f'Building model')
    config.num_labels = 1

    model = Jane(config)
    model = model.to(device)

    summary(model, input_size=(16, config.feature_size))

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    # lr = torch.optim.lr_scheduler.ExponentialLR(opt, config.lr_decay)
    lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 
            factor=config.lr_decay,
            patience=3
        )
    met = METResolution()

    # model, opt = amp.initialize(model, opt, opt_level='O1')

    if not os.path.exists(config.plot):
        os.makedirs(config.plot)

    for e in range(config.n_epochs):
        logger.info(f'Epoch {e}: Start')
        current_lr = [group['lr'] for group in opt.param_groups][0]
        logger.info(f'Epoch {e}: Current LR = {current_lr}')

        model.train()
        met.reset()
        avg_loss_tensor = 0
        for n_batch, batch in enumerate(tqdm(dl, total=int(len(ds)/config.batch_size), leave=False)):
            if n_batch % 1000 == 0:
                logger.debug(n_batch)
            x = torch.Tensor(batch[0]).to(device)
            m = torch.LongTensor(batch[2]).to(device)
            gm = torch.Tensor(batch[5]).to(device)
            
            opt.zero_grad()
            loss, met_pred = model(x, y=gm, mask=m)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            avg_loss_tensor += loss 

            x = batch[0] 
            puppi = batch[4]
            gm = batch[5] 
            pt = x[:,:,0]
            phi = x[:,:,2]
            met.compute(
                    pt, phi,
                    puppi, 
                    gm,
                    t2n(met_pred)
                )

        avg_loss_tensor /= n_batch
        lr.step(avg_loss_tensor)

        plot_path = f'{config.plot}/resolution_{e:03d}'
        ress = met.plot(plot_path)
        model_res = ress['model']
        puppi_res = ress['puppi']

        logger.info(f'Epoch {e}: MODEL MET error = {model_res[0]} +/- {model_res[1]}, loss = {t2n(avg_loss_tensor)}') 
        logger.info(f'Epoch {e}: PUPPI MET error = {puppi_res[0]} +/- {puppi_res[1]}') 

        torch.save(model.state_dict(), snapshot.get_path(f'model_weights_epoch{e}.pt'))
