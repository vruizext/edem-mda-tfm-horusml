import random
import os
import numpy as np
from fastai.vision.all import *

def seed_everything (seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed (seed)
    torch.manual_seed (seed)
    torch.cuda.manual_seed (seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed, reproducible=True)


def generate_split(df, split_counts, random_seed):
    idx = np.array([])
    for k, count in list(split_counts.items()):
        idx = np.concatenate([idx, df[df['FEVI10'] == k].sample(count, random_state=random_seed).index], axis=0)

    return idx