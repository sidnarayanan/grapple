import torch
from torch import nn 
from .utils import t2n 


class Metrics(object):
    def __init__(self, device):
        self.loss_calc = nn.CrossEntropyLoss(
                ignore_index=-1, 
                # weight=torch.FloatTensor([1, 5]).to(device)
            )
        self.reset()

    def reset(self):
        self.loss = 0 
        self.acc = 0 
        self.pos_acc = 0
        self.neg_acc = 0
        self.n_pos = 0
        self.n_particles = 0
        self.n_steps = 0

    def compute(self, yhat, y):
        # yhat = [batch, particles, labels]; y = [batch, particles]
        loss = self.loss_calc(yhat.view(-1, yhat.shape[-1]), y.view(-1))
        self.loss += t2n(loss).mean()

        mask = (y != -1)
        n_particles = t2n(mask.sum())

        pred = torch.argmax(yhat, dim=-1) # [batch, particles]
        acc = t2n((pred == y)[mask].sum()) / n_particles 
        self.acc += acc

        n_pos = t2n((y == 1).sum())
        pos_acc = t2n((pred == y)[y == 1].sum()) / n_pos
        self.pos_acc += pos_acc
        neg_acc = t2n((pred == y)[y == 0].sum()) / (n_particles - n_pos)
        self.neg_acc += neg_acc

        self.n_pos += n_pos
        self.n_particles += n_particles

        self.n_steps += 1

        if self.n_steps % 50 == 0 and False:
            print(t2n(y[0])[:10])
            print(t2n(pred[0])[:10])
            print(t2n(yhat[0])[:10, :])

        return loss, acc

    def mean(self):
        return ([x / self.n_steps 
                 for x in [self.loss, self.acc, self.pos_acc, self.neg_acc]]
                + [self.n_pos / self.n_particles])
