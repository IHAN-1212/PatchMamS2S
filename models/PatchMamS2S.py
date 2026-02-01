import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from modules.Mamba2Local import Mamba2Simple
class EmbLayer(nn.Module):

    def __init__(self, patch_len, patch_step, in_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        patch_num = int((in_len - patch_len) / patch_step + 1)
        self.d_model = d_model // patch_num
        self.ff = nn.Linear(patch_len, self.d_model)
        self.flatten = nn.Flatten(start_dim=-2)

        self.ff_1 = nn.Linear(self.d_model * patch_num, d_model)

    def forward(self, x):
        B, V, L = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        x = self.ff(x)
        x = self.flatten(x)

        x = self.ff_1(x)
        return x
class Emb(nn.Module):

    def __init__(self, in_len, d_model, patch_len=[48, 24, 12, 6]):
        super().__init__()
        patch_step = patch_len
        d_model = d_model//4
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, in_len, d_model)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, in_len, d_model)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, in_len, d_model)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, in_len, d_model)

    def forward(self, x):
        s_x1 = self.EmbLayer_1(x)
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        s_out = torch.cat([s_x1, s_x2, s_x3, s_x4], -1) #(B, V, 4*D)
        return s_out
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)

        x = self.avg(x)
        return x
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
class MTSModel(nn.Module):

    def __init__(self, args):
        super(MTSModel, self).__init__()
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.d_model = args.d_model
        # self.output_attention = args.output_attention
        self.use_norm = args.use_norm

        decomp = args.k # 13

        self.decompsition1 = series_decomp(decomp)
        self.decompsition2 = series_decomp(decomp)
        self.decompsition3 = series_decomp(decomp)
        self.decompsition4 = series_decomp(decomp)
        # Embedding
        # self.emb = Emb(args.in_len, args.d_model)

        patch_len = [48, 24, 12, 6]
        patch_step = patch_len
        self.per_d_model = self.d_model
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, self.in_len, self.per_d_model)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, self.in_len, self.per_d_model)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, self.in_len, self.per_d_model)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, self.in_len, self.per_d_model)
        # End Embedding

        exp_weight_config = {
            "base": (0.5, 0.25, 0.15, 0.1),  # 基准组
            "uniform": (0.25, 0.25, 0.25, 0.25),  # 均匀对照组
            "small": (1.0, 0.0, 0.0, 0.0),  # 极端小权重组（仅weight_a）
            "large": (0.0, 0.0, 0.0, 1.0),  # 极端大权重组（仅weight_d）
            "reverse": (0.1, 0.15, 0.25, 0.5),  # 反向梯度组
            "linear": (0.4, 0.3, 0.2, 0.1),  # 线性梯度组
            "variant": (0.6, 0.2, 0.15, 0.05),  # 梯度变体组
            "tradition": (1.0, 1.0, 1.0, 1.0)  # 传统融合对照组
        }
        exp_key = args.ma_type
        # 定义融合比例
        try:
            self.weight_a, self.weight_b, self.weight_c, self.weight_d = exp_weight_config[exp_key]
        except KeyError:
            raise ValueError(f"无效实验关键词！可选关键词：{list(exp_weight_config.keys())}")

        # self.seasonal_layers = Encoder(args.d_model, args.data_dim)
        # self.trend_layers = Encoder(args.d_model, args.data_dim)
        self.s2s = Encoder(args)

        self.projector = nn.Linear(args.d_model, args.out_len, bias=True)


    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        x = x_enc.permute(0, 2, 1)
        # Embedding
        # x = self.emb(x)
        s_x1 = self.EmbLayer_1(x)
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        # s_out = torch.cat([s_x1, s_x2, s_x3, s_x4], -1)
        # x = s_out
        # End Embedding

        s1, t1 = self.decompsition1(s_x1)
        s2, t2 = self.decompsition2(s_x2)
        s3, t3 = self.decompsition3(s_x3)
        s4, t4 = self.decompsition4(s_x4)

        seasonal_init = self.weight_a * s4 + self.weight_b * s3 + self.weight_c * s2 + self.weight_d * s1
        trend_init = self.weight_a * t1 + self.weight_b * t2 + self.weight_c * t3 + self.weight_d * t4


        # seasonal_init = self.seasonal_layers (seasonal_init)
        # trend_init = self.trend_layers(trend_init)
        # x = seasonal_init + trend_init

        x = self.s2s(seasonal_init, trend_init)

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


from einops import repeat
class Encoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        c_d_state = args.d_state
        c_headdim = args.headdim

        self.Mamba2_enc = Mamba2Simple(d_model=args.d_model, d_state=c_d_state, headdim=c_headdim,learnable_init_states=False)
        self.Mamba2_dec = Mamba2Simple(d_model=args.d_model, d_state=c_d_state, headdim=c_headdim,learnable_init_states=True)


    def forward(self, seasonal, trend):
        _, hidden = self.Mamba2_enc(trend)
        y, _ = self.Mamba2_dec(seasonal,local_init=hidden)
        return y
