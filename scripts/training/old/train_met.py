#!/usr/bin/env python3
from grapple.utils import * 

p = ArgumentParser()
p.add_args(
    '--train_dataset_pattern', 
    '--test_dataset_pattern', 
    '--output', ('--n_epochs', p.INT),
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--num_hidden_layers', p.INT), ('--batch_size', p.INT),
    ('--mask_charged', p.STORE_TRUE), ('--lr', {'type': float}),
    ('--lr_schedule', p.STORE_TRUE), '--plot',
    ('--min_met', p.FLOAT),
    ('--num_max_files', p.INT),
    ('--dr_adj', p.FLOAT),
)
config = p.parse_args()


from grapple.data import METDataset, DataLoader
from grapple.metrics import * 
from grapple.model import * 

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

    logger.info(f'Building model')
    config.num_labels = 1

    model = Oskar(config)
    model = model.to(device)

    #summary(model, input_size=(16, config.feature_size))

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr = torch.optim.lr_scheduler.ExponentialLR(opt, config.lr_decay)
    # lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         opt, 
    #         factor=config.lr_decay,
    #         patience=3
    #     )
    met = METResolution()

    # model, opt = amp.initialize(model, opt, opt_level='O1')

    if not os.path.exists(config.plot):
        os.makedirs(config.plot)

    for e in range(config.n_epochs):
        logger.info(f'Epoch {e}: Start')
        current_lr = [group['lr'] for group in opt.param_groups][0]
        logger.info(f'Epoch {e}: Current LR = {current_lr}')

        if e < 3:
            training_mode = 0
        elif e < 6:
            training_mode = 1
        else:
            training_mode = 2

        training_mode = 2

        train_ds = METDataset(config.train_dataset_pattern, config, 10, 15, training_mode)
        test_ds = METDataset(config.test_dataset_pattern, config, 10, 15, training_mode)
        train_dl = DataLoader(train_ds, batch_size=config.batch_size, collate_fn=METDataset.collate_fn)
        test_dl = DataLoader(test_ds, batch_size=config.batch_size, collate_fn=METDataset.collate_fn)

        model.train()
        met.reset()
        train_avg_loss_tensor = 0
        for n_batch, batch in enumerate(tqdm(train_dl, total=int(len(train_ds)/config.batch_size), leave=False)):
            x = torch.Tensor(batch[0]).to(device)
            m = torch.LongTensor(batch[1]).to(device)
            gm = torch.Tensor(batch[2]).to(device)

            opt.zero_grad()
            loss, met_pred = model(x, y=gm, mask=m)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            train_avg_loss_tensor += loss 

        train_avg_loss_tensor /= n_batch
        lr.step()

        model.eval()
        met.reset()
        test_avg_loss_tensor = 0
        for n_batch, batch in enumerate(tqdm(test_dl, total=int(len(test_ds)/config.batch_size), leave=False)):
            x = torch.Tensor(batch[0]).to(device)
            m = torch.LongTensor(batch[1]).to(device)
            gm = torch.Tensor(batch[2]).to(device)

            with torch.no_grad():
                loss, met_pred = model(x, y=gm, mask=m)
            test_avg_loss_tensor += loss 

            gm = test_ds.unstandardize_met(batch[2]) 
            pf = test_ds.unstandardize_met(batch[3])
            met_pred = test_ds.unstandardize_met(t2n(met_pred))
            met.compute(
                    pf, 
                    gm,
                    met_pred
                )

        # print(gm)
        # print(met_pred)

        test_avg_loss_tensor /= n_batch

        plot_path = f'{config.plot}/resolution_{e:03d}'
        ress = met.plot(plot_path)
        model_res = ress['model']
        puppi_res = ress['puppi']

        logger.info(f'Epoch {e}: MODEL MET error = {model_res[0]} +/- {model_res[1]}')
        logger.info(f'Epoch {e} train loss = {t2n(train_avg_loss_tensor)}, test loss = {t2n(test_avg_loss_tensor)}') 
        logger.info(f'Epoch {e}: PUPPI MET error = {puppi_res[0]} +/- {puppi_res[1]}') 

        torch.save(model.state_dict(), snapshot.get_path(f'model_weights_epoch{e}.pt'))
