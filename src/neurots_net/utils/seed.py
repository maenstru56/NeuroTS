from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    os.environ["PYTHONHASHSEED"] = str(value)
