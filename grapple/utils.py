from argparse import SUPPRESS, ArgumentParser as _AP
from loguru import logger
import os
import sys
import time
import yaml


class Opt(dict):
    def __init__(self, *args, **kwargs):
        super(Opt, self).__init__()
        for a in args:
            if isinstance(a, dict):
                self.update(a)
        self.update(kwargs)

    def __add__(self, other):
        return Opt(self, other)

    def __iadd__(self, other):
        self.update(other)
        return self 


class ArgumentParser(_AP):
    STORE_TRUE = Opt({'action':'store_true'})
    STORE_FALSE = Opt({'action':'store_false'})
    MANY = Opt({'nargs':'+'})
    INT = Opt({'type': int})

    class Namespace(object):
        def __init__(self):
            pass

        def save_to(self, path):
            yaml.dump({k:getattr(self, k) for k in vars(self)},
                      open(path, 'w'),
                      default_flow_style=True)

        def __str__(self):
            return str({k:getattr(self, k) for k in vars(self)})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().add_argument('-c', '--config', nargs='+', default=[])

    def add_arg(self, *args, **kwargs):
        if 'default' in kwargs:
            logger.error(f'default is not allowed in ArgumentParser')
            raise RuntimeError()
        return super().add_argument(*args, **kwargs)

    def add_args(self, *args):
        for a in args:
            if type(a) == tuple:
                self.add_arg(a[0], **a[1])
            else:
                self.add_arg(a)

    def parse_args(self, *args, **kwargs):
        cmd_line_args = super().parse_args(*args, **kwargs)
        args = ArgumentParser.Namespace()
        for conf in cmd_line_args.config:
            payload = yaml.safe_load(open(conf, 'r'))
            for k,v in payload.items():
                setattr(args, k, v)
                logger.debug(f'Config {conf} : {k} -> {v}')
        for k in vars(cmd_line_args):
            v = getattr(cmd_line_args, k)
            if v is None:
                continue
            setattr(args, k, v)
            logger.debug(f'Command line : {k} -> {v}')
        self.args = args

        return args 


class Snapshot(object):
    def __init__(self, base_path, args):
        self.path = os.path.join(base_path, time.strftime("%Y_%m_%d_%H_%M_%S"))
        logger.info(f'Snapshot placed at {self.path}')
        os.makedirs(self.path)
        self.args = args
        args.save_to(self.get_path('args.yaml'))
        logger.remove()
        logger.add(sys.stderr, level='INFO')
        logger.add(self.get_path('snapshot.log'), level='DEBUG')

    def get_path(self, filename):
        return os.path.join(self.path, filename)


def t2n(t):
    return t.to('cpu').detach().numpy()
