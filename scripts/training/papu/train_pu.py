#!/usr/bin/env python3
from grapple import utils

p = utils.ArgumentParser()
p.add_args(
    '--dataset_pattern', '--output', ('--n_epochs', p.INT),
    '--checkpoint_path',
    ('--met_constraint', p.FLOAT),
    ('--num_global_objects', p.INT),
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--label_size', p.INT), ('--num_hidden_layers', p.INT), ('--batch_size', p.INT),
    ('--num_hidden_groups', p.INT), ('--inner_group_num', p.INT),
    ('--mask_charged', p.STORE_TRUE), ('--lr', {'type': float}),
    ('--attention_band', p.INT),
    ('--epoch_offset', p.INT),
    ('--from_snapshot'),
    ('--lr_schedule', p.STORE_TRUE), '--plot',
    ('--pt_weight', p.INT), ('--num_max_files', p.INT),
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
from apex.optimizers import FusedAdam, FusedLAMB
from functools import partial
from glob import glob
import re
from transformers.optimization import get_linear_schedule_with_warmup as linear_sched


def scale_fn(c, decay):
    return decay ** c


def check_mem_stats(device=None):
    cached = torch.cuda.memory_cached(device) // 1.e9
    allocated = torch.cuda.memory_allocated(device) // 1.e9
    logger.info(f'Device {device} memory: {allocated} GB ({cached} GB) allocated (cached)')


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

    logger.info(f'Computational batch size is {config.batch_size}, accumulated over {config.grad_acc} steps')

    logger.info(f'Reading dataset at {config.dataset_pattern}')
    num_workers = 8
    ds = PapuDataset(config)
    dl = DataLoader(ds, batch_size=config.batch_size, 
                    collate_fn=PapuDataset.collate_fn,
                    num_workers=num_workers, pin_memory=True)
    batches_per_epoch = len(ds) // config.batch_size * num_workers
    steps_per_epoch = batches_per_epoch // config.grad_acc

    config.epoch_offset = 0 

    logger.info(f'Building model')

    def load_checkpoint(path):
        existing_checkpoints = glob(snapshot.get_path(path))
        if existing_checkpoints:
            ckpt = sorted(existing_checkpoints)[-1]
            config.from_snapshot = ckpt
            epoch = int(re.sub(r'.*epoch', '', re.sub(r'\.pt$', '', ckpt)))
            config.epoch_offset = epoch
            
            if config.lr_policy != 'linear':
                lr_steps = epoch // (4 if config.lr_policy == 'cyclic' else 1)
                config.lr *= (config.lr_decay ** lr_steps)
            return True 
        else:
            return False

    if config.from_snapshot is None:
        loaded = load_checkpoint('model_weights_best_epoch*pt')
        if not loaded:
            # if best doesn't exist, take the latest
            loaded = load_checkpoint('model_weights_epoch*pt')

    model = Bruno(config)
    if config.from_snapshot is not None:
        state_dicts = torch.load(config.from_snapshot)
        model.load_state_dict(state_dicts['model'])

        logger.info(f'Model ckpt {config.from_snapshot} loaded.')

    model = model.to(device)

    # opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    opt = FusedLAMB(model.parameters(), lr=config.lr) 

    if config.from_snapshot is not None:
        state_dicts = torch.load(config.from_snapshot)
        opt.load_state_dict(state_dicts['opt'])

    if config.lr_policy == 'exp' or config.lr_policy is None:
        lr = torch.optim.lr_scheduler.ExponentialLR(opt, config.lr_decay)
    elif config.lr_policy == 'cyclic':
        lr = torch.optim.lr_scheduler.CyclicLR(opt, 0, config.lr, step_size_up=steps_per_epoch*2,
                                               scale_fn=partial(scale_fn, decay=config.lr_decay),
                                               cycle_momentum=False)
    elif config.lr_policy == 'linear':
        lr = linear_sched(opt, num_warmup_steps=steps_per_epoch*2, num_training_steps=128*steps_per_epoch,
                          last_epoch=(config.epoch_offset*steps_per_epoch)-1) # 'epoch' here refers to batch for us

    # if config.from_snapshot is not None:
    #     state_dicts = torch.load(config.from_snapshot)
    #     opt.load_state_dict(state_dicts['lr'])

    if config.attention_band is not None:
        model, opt = amp.initialize(model, opt, opt_level='O1')


    # lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         opt, 
    #         factor=config.lr_decay,
    #         patience=3
    #     )
    metrics = PapuMetrics(config.beta, config.met_constraint)
    metrics_puppi = PapuMetrics()
    metres = ParticleMETResolution()

    if torch.cuda.device_count() > 1:
        logger.info(f'Distributing model across {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    if not os.path.exists(config.plot):
        os.makedirs(config.plot)

    if config.epoch_offset is not None:
        min_epoch = config.epoch_offset+1
    else:
        min_epoch = 0

    if config.pt_weight is None:
        config.pt_weight = 0

    for e in range(min_epoch, config.n_epochs+min_epoch):
        logger.info(f'Epoch {e}: Start')
        current_lr = [group['lr'] for group in opt.param_groups][0]
        logger.info(f'Epoch {e}: Current LR = {current_lr}')
        logger.info(f'Epoch {e}: N_particles = {ds.n_particles}')

        # model.unfreeze_all()

        # if e < 13:
        #     # only fit PU
        #     model.freeze_met()
        #     metrics.met_loss_weight = metrics_puppi.met_loss_weight = 0.
        # elif e < 25:
        #     # only fit MET, based on a frozen encoder
        #     model.freeze_pu()
        #     metrics.met_loss_weight = metrics_puppi.met_loss_weight = 1

        model.train()
        metrics.reset()
        metrics_puppi.reset()
        metres.reset()

        best_loss = np.inf

        avg_loss_tensor = 0
        # tqdm = lambda x, **kwargs: x
        opt.zero_grad()
        ready_for_lr = False
        for n_batch, batch in enumerate(tqdm(dl, total=batches_per_epoch)):
            sparse.VERBOSE = (n_batch == 0)

            x = to_t(batch['x'])
            y = to_t(batch['y'])
            m = to_lt(batch['mask'])
            p = to_t(batch['puppi'])
            qm = to_lt(batch['mask'] & batch['neutral_mask'])
            cqm = to_lt(batch['mask'] & ~batch['neutral_mask'])
            genmet = batch['genmet'][:, 0]
            genmetphi = batch['genmet'][:, 1]

            if config.pt_weight == 0:
                weight  = None
            else:
                weight = x[:, :, 0] / x[:, 0, 0].reshape(-1, 1)
                weight = weight ** config.pt_weight

            if True or e < 3:
                loss_mask = m 
            else:
                loss_mask = qm

            yhat = model(x, attn_mask=m)
            if not config.beta:
                yhat = torch.sigmoid(yhat)
            else:
                yhat = torch.relu(yhat)
            loss, _ = metrics.compute(yhat, y, w=weight, m=loss_mask, plot_m=qm, x=x)
            loss /= config.grad_acc
            if config.attention_band is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (n_batch+1) % config.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                opt.zero_grad()
                ready_for_lr = True

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
                           gm=genmet,
                           gmphi=genmetphi)

            if config.lr_policy == 'cyclic':
                if ready_for_lr:
                    lr.step()
                    # current_lr = [group['lr'] for group in opt.param_groups][0]
                    # logger.info(f'Epoch {e}: Step {n_batch}: Current LR = {current_lr}')

            if e == 0 & n_batch == 100:
                check_mem_stats(device)

        check_mem_stats(device)

        avg_loss_tensor /= n_batch
        if config.lr_policy != 'cyclic':
            if ready_for_lr:
                lr.step()

        plot_path = f'{config.plot}/resolution_{e:03d}'

        metrics.plot(plot_path + '_model')
        metrics_puppi.plot(plot_path + '_puppi')
        resolution = metres.plot(plot_path + '_met')

        avg_loss, avg_acc, avg_posacc, avg_negacc, avg_posfrac = metrics.mean()
        logger.info(f'Epoch {e}: Average fraction of hard particles = {avg_posfrac}')
        logger.info(f'Epoch {e}: MODEL:')
        logger.info(f'Epoch {e}: Loss = {avg_loss}; Accuracy = {avg_acc}')
        logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')

        is_best = False 
        if avg_loss < best_loss:
            best_loss = avg_loss
            is_best = True
        
        avg_loss, avg_acc, avg_posacc, avg_negacc, _ = metrics_puppi.mean()
        logger.info(f'Epoch {e}: PUPPI:')
        logger.info(f'Epoch {e}: Loss = {avg_loss}; Accuracy = {avg_acc}')
        logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')

        logger.info(f'Epoch {e}: Model MET err = {resolution["model"][0]} +/- {resolution["model"][1]}')
        logger.info(f'Epoch {e}: Puppi MET err = {resolution["puppi"][0]} +/- {resolution["puppi"][1]}')

        model_to_save = model.module if torch.cuda.device_count() > 1 else model
        state_dicts = {'model': model_to_save.state_dict(),
                       'opt': opt.state_dict(),
                       'lr': lr.state_dict()}

        torch.save(state_dicts, snapshot.get_path(f'model_weights_epoch{e:06d}.pt'))
        if is_best:
            torch.save(state_dicts, snapshot.get_path(f'model_weights_best_epoch{e:06d}.pt'))

        # ds.n_particles = min(2000, ds.n_particles + 50)
