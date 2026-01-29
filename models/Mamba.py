import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mamba_ssm import Mamba

class MTSModel(nn.Module):

    def __init__(self, args):
        super(MTSModel, self).__init__()
        self.in_len = args.in_len
        self.out_len = args.out_len
        # self.output_attention = args.output_attention
        self.use_norm = args.use_norm

        self.mamba = Mamba(d_model=args.d_model)

        self.projectorin = nn.Linear(args.in_len, args.d_model, bias=True)
        self.projector = nn.Linear(args.d_model, args.out_len, bias=True)


    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        x = x_enc.permute(0, 2, 1)
        x = self.projectorin(x)
        x = self.mamba(x)
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.out_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.out_len, 1))

        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.out_len:, :]  # [B, L, D]
