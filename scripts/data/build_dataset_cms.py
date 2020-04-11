#!/usr/bin/env python3

from grapple.data.cms import cms_to_grapple
from grapple.utils import * 

import numpy as np
import os
from tqdm import tqdm

from loguru import logger


parser = ArgumentParser()
parser.add_args(
    ('--infiles', {'nargs': '+', 'required': True}),
    ('--outdir', {'required': True}),
)
args = parser.parse_args()

if __name__ == '__main__':
    snapshot = Snapshot(args.outdir, args)
    for f in tqdm(args.infiles[:50]):
        try:
            cms_to_grapple(
                f, snapshot.get_path(os.path.split(f)[-1].replace('root', 'npz'))
            )
        except Exception as e:
            logger.debug(f)
            logger.error(e)
