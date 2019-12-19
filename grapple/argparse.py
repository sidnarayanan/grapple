from argparse import ArgumentParser as _AP
from loguru import logger
import yaml


class Namespace(object):
    def __init__(self):
        pass

    def save_to(self, path):
        yaml.dump({k:getattr(self, k) for k in vars(self)},
                  open(path, 'w'),
                  default_flow_style=True)


class ArgumentParser(_AP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('-c', '--config', nargs='+', default=[])

    def parse_args(self, *args, **kwargs):
        cmd_line_args = super().parse_args(*args, **kwargs)
        args = Namespace()
        for conf in cmd_line_args.config:
            payload = yaml.safe_load(open(conf, 'r'))
            for k,v in payload.items():
                setattr(args, k, v)
                logger.debug(f'Config {conf} : {k} -> {v}')
        for k in vars(cmd_line_args):
            v = getattr(cmd_line_args, v)
            setattr(args, k, v)
            logger.debug(f'Command line : {k} -> {v}')
        self.args = args

        return args 
