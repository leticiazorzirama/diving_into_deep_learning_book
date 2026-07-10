# 2.1. Data Manipulation
# 2.1.1. Getting Started

import torch
import numpy

x = torch.arange(12, dtype=torch.float32)
x.numel()
x.shape
x

X = x.reshape(3,4)
X

torch.zeros((2,3,4))
torch.ones((2,3,4))
torch.randn(3,4)
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
