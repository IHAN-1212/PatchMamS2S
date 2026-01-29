import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, -1, keepdim=True))
        x_norm = x / (rms + self.epsilon)
        return self.gamma * x_norm

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias