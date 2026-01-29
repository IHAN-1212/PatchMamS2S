import argparse
import os
import torch
import random
import numpy as np
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import string_split
import time
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime


if __name__ == '__main__':
    #fix_random_seed
    seed_value=2021
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    #start
    parser = argparse.ArgumentParser(description='MTSForecasting')
    #default
    parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--seed', type=int, default=2021, help='experiments seed')
    #gpu
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
    #datasets
    parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2',help='train/val/test split, can be ratio or number')
    #base
    parser.add_argument('--model', type=str, required=True, default='Crossformer', help='MTS Model')
    parser.add_argument('--in_len', type=int, default=96, help='input MTS length (T)')
    parser.add_argument('--out_len', type=int, default=96, help='output MTS length (tau)')
    #Hyper-params-Base
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer in (adam cadamw)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    #Visualization
    parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

    #Hyper-params-Share
    parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
    parser.add_argument('--d_state', type=int, default=1, help='Mamba\' state')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP')
    #Crossformer-Only
    parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
    parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
    parser.add_argument('--factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (N)')
    parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)
    #Crossmamba-Only
    parser.add_argument('--t_cycle', type=int, default=6, help='segment length (t_cycle)')
    #MamS2S-Only
    parser.add_argument('--m', type=int, default=96, help='takens\' m')
    parser.add_argument('--k', type=int, default=13, help='takens\' k')
    parser.add_argument('--headdim', type=int, default=8, help='parser of Mamba2')
    #PatchMS2S-Only
    parser.add_argument('--s2s_layers', type=int, default=3, help='num of MS2S layers')

    #PatchMLP-Only
    parser.add_argument('--use_norm', type=bool, default=True, help='use custom normalization')

    # Patching
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

    # Moving Average
    parser.add_argument('--ma_type', type=str, default='ema', help='reg, ema, dema')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha')
    parser.add_argument('--beta', type=float, default=0.3, help='beta')

    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')

    #end
    args = parser.parse_args()
    #gup_setting
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        print(args.gpu)
    #datasets_setting
    data_parser = {
        'ETTh1':{'data':'ETTh1.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
        'etth1':{'data':'ETTh1.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
        'ETTh2':{'data':'ETTh2.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
        'etth2':{'data':'ETTh2.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
        'ETTm1':{'data':'ETTm1.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
        'ettm1':{'data':'ETTm1.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
        'ETTm2':{'data':'ETTm2.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
        'ettm2':{'data':'ETTm2.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
        'WTH':{'data':'WTH.csv', 'data_dim':12, 'split':[28*30*24, 10*30*24, 10*30*24]},
        'wth':{'data':'WTH.csv', 'data_dim':12, 'split':[28*30*24, 10*30*24, 10*30*24]},
        'ILI':{'data':'national_illness.csv', 'data_dim':7, 'split':[0.7, 0.1, 0.2]},
        'ili':{'data':'national_illness.csv', 'data_dim':7, 'split':[0.7, 0.1, 0.2]},
        'ECL': {'data': 'ECL.csv', 'data_dim': 321, 'split': [0.7, 0.1, 0.2]},
        'ecl': {'data': 'ECL.csv', 'data_dim': 321, 'split': [0.7, 0.1, 0.2]},
        'Weather': {'data': 'weather.csv', 'data_dim': 21, 'split': [0.7, 0.1, 0.2]},
        'weather': {'data': 'weather.csv', 'data_dim': 21, 'split': [0.7, 0.1, 0.2]},
        'Traffic': {'data': 'traffic.csv', 'data_dim': 862, 'split': [0.7, 0.1, 0.2]},
        'traffic': {'data': 'traffic.csv', 'data_dim': 862, 'split': [0.7, 0.1, 0.2]},
        'Exchange': {'data': 'exchange_rate.csv', 'data_dim': 8, 'split': [0.7, 0.1, 0.2]},
        'exchange': {'data': 'exchange_rate.csv', 'data_dim': 8, 'split': [0.7, 0.1, 0.2]},
        'Solar': {'data': 'solar_AL.txt', 'data_dim': 137, 'split': [0.7, 0.1, 0.2]},
        'solar': {'data': 'solar_AL.txt', 'data_dim': 137, 'split': [0.7, 0.1, 0.2]},
        'Australia': {'data': 'Australia.csv', 'data_dim': 6, 'split': [0.7, 0.1, 0.2]},
        'EP': {'data': 'electric_price.csv', 'data_dim': 63, 'split': [0.7, 0.1, 0.2]}
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.data_dim = data_info['data_dim']
        args.data_split = data_info['split']
    else:
        args.data_split = string_split(args.data_split)
    #end_args
    print('Args in experiment:')
    print(args)
    #main
    Exp = Exp_Long_Term_Forecast
    for ii in range(args.itr):
        # setting record of experiments
        # 定义模型设置和参数映射

        # 获取当前日期时间
        current_datetime = datetime.now()
        # 自定义格式输出（更易读）
        formatted_datetime = current_datetime.strftime("%m%d_%H%M")
        print("当前日期时间（自定义格式）：", formatted_datetime)
        model_config = {
            'Crossformer': {
                'format': '{}_{}_{}_{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_itr{}',
                'params': lambda args, ii: [
                    args.model, args.data, args.in_len, args.out_len,
                    args.seg_len, args.win_size, args.factor, args.d_model, args.n_heads, args.e_layers,
                    ii
                ]
            },
            'xPatch': {
                'format': '{}_{}_{}_{}_bs{}_itr{}',
                'params': lambda args, ii: [
                    args.model, args.data, args.in_len, args.out_len,
                    args.batch_size,
                    ii
                ]
            },
            'PatchMLP': {
                'format': '{}_{}_{}_{}_dm{}_el{}_itr{}',
                'params': lambda args, ii: [
                    args.model, args.data, args.in_len, args.out_len,
                    args.d_model, args.e_layers,
                    ii
                ]
            },
            'PatchMamS2S': {
                'format': '{}_paper_{}_{}_{}_dm{}_{}_{}_lr{}_decomp{}_itr{}_time{}_{}',
                'params': lambda args, ii: [
                    args.model, args.data, args.in_len, args.out_len,
                    args.d_model, args.d_state, args.headdim,
                    args.lradj, args.k,
                    ii, formatted_datetime, args.ma_type
                ]
            },
            'Mamba': {
                'format': '{}_{}_{}_{}_dm{}_lr{}_itr{}_time{}',
                'params': lambda args, ii: [
                    args.model, args.data, args.in_len, args.out_len,
                    args.d_model,
                    args.lradj,
                    ii, formatted_datetime
                ]
            },
            'Crossmamba': {
                'format': '{}_{}_{}_{}_cyc{}_dm{}_ds{}_hd{}_op{}_sl{}_itr{}',
                'params': lambda args, ii: [
                    args.model, args.data, args.in_len, args.out_len,
                    args.t_cycle, args.d_model, args.d_state, args.headdim, args.optimizer, args.s2s_layers,
                    ii
                ]
            },
        }

        # 根据模型名称选择相应的设置格式
        if args.model in model_config:
            config = model_config[args.model]
            setting = config['format'].format(*config['params'](args, ii))
        else:
            raise ValueError(f"不支持的模型: {args.model}")

        exp = Exp(args) # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, args.save_pred)
