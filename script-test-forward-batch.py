import architecture
device = 'cuda'
hamster = architecture.Hamster(device)
hamster.initiateLayer()

import torch
x = torch.randn(12, 3, 128, 128).to(device)
r = torch.randn(12, 3, 128, 128).to(device)
f = torch.randn(12, 3, 128, 128).to(device)
t = torch.randint(0, 1501, (12,)).to(device)
y = hamster(x, t)

