#!/usr/bin/env python3
from grapple.utils import * 

p = ArgumentParser()
p.add_args(
    '--dataset_pattern', '--output', ('--n_epochs', p.INT),
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--label_size', p.INT), ('--num_hidden_layers', p.INT), ('--batch_size', p.INT),
    ('--mask_charged', p.STORE_TRUE), ('--lr', {'type': float}),
    ('--lr_schedule', p.STORE_TRUE)
)
config = p.parse_args()


from grapple.data import PUDataset, DataLoader
from grapple.metrics import Metrics
from grapple.model import Joe

from tqdm import tqdm, trange
from loguru import logger
import torch
from torch import nn 
from torch.utils.data import RandomSampler


if __name__ == '__main__':
    snapshot = Snapshot(config.output, config)
    logger.info(f'Saving output to {snapshot.path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    logger.info(f'Reading dataset at {config.dataset_pattern}')
    ds = PUDataset(config)
    dl = DataLoader(ds, batch_size=config.batch_size, collate_fn=PUDataset.collate_fn)

    logger.info(f'Building model')
    model = Joe(config)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    # lr = torch.optim.lr_scheduler.ExponentialLR(opt, config.lr_decay)
    lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 
            factor=config.lr_decay,
            patience=3
        )
    metrics = Metrics(device)

    for e in range(config.n_epochs):
        logger.info(f'Epoch {e}: Start')
        current_lr = [group['lr'] for group in opt.param_groups][0]
        logger.info(f'Epoch {e}: Current LR = {current_lr}')

        model.train()
        metrics.reset()
        avg_loss_tensor = 0
        for n_batch, batch in enumerate(dl):
            x = torch.Tensor(batch[0]).to(device)
            y = torch.LongTensor(batch[1]).to(device)
            m = torch.LongTensor(batch[2]).to(device)
            
            opt.zero_grad()
            yhat = model(x, mask=m)
            loss, _ = metrics.compute(yhat, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            avg_loss_tensor += loss 

        avg_loss_tensor /= n_batch
        lr.step(avg_loss_tensor)

        avg_loss, avg_acc, avg_posacc, avg_negacc, avg_posfrac = metrics.mean()
        logger.info(f'Epoch {e}: Loss = {avg_loss}; Accuracy = {avg_acc}')
        logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')
        logger.info(f'Epoch {e}: Average fraction of hard particles = {avg_posfrac}')
        torch.save(model.state_dict(), snapshot.get_path(f'model_weights_epoch{e}.pt'))
