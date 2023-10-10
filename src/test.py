import torch
import torch.nn as nn

v = torch.tensor([0], dtype=torch.float)
m = nn.Linear(1, 10)
m(v)
