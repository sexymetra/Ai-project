import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#하이퍼 파라미터
CFG = {
    'EPOCHS':10,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':128,
    'SEED':41
}
