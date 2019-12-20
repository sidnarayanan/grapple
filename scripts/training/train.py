#!/usr/bin/env python3
from grapple.utils import * 

p = ArgumentParser()
p.add_args(
    '--dataset_pattern', '--output', ('--n_epochs', p.INT),
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--label_size', p.INT), ('--num_hidden_layers', p.INT), ('--batch_size', p.INT)
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

    opt = torch.optim.Adam(model.parameters())
    metrics = Metrics()

    for e in range(config.n_epochs):
        logger.info(f'Epoch {e}: Start')

        model.train()
        metrics.reset()
        for batch in dl:
            x = torch.Tensor(batch[0]).to(device)
            y = torch.LongTensor(batch[1]).to(device)
            m = torch.LongTensor(batch[2]).to(device)
            
            opt.zero_grad()
            yhat = model(x, mask=m)
            loss, _ = metrics.compute(yhat, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            opt.step()

        avg_loss, avg_acc, avg_posacc, avg_negacc = metrics.mean()
        logger.info(f'Epoch {e}: Loss = {avg_loss}; Accuracy = {avg_acc}')
        logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')

