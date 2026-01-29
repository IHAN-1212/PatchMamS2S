from data.data_loader import Dataset_MTS, Dataset_Solar
from exp.exp_basic import Exp_Basic

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.efficiency_statistics import count_parameters, get_model_profile, measure_inference_time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
import json
import pickle
from tqdm import tqdm
from torchsummary import summary
from thop import profile
from utils.c_adamw import AdamW

import warnings
warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
    
    def _build_model(self):
        model=self.model_dict[self.args.model].MTSModel(
            self.args
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
        if args.data == 'Solar':
            data_set = Dataset_Solar(
                flag=flag,
                size=[args.in_len, args.out_len],
            )
        else:
            data_set = Dataset_MTS(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.in_len, args.out_len],
                data_split = args.data_split,
            )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer == 'cadamw':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        custom_params = count_parameters(self.model)
        sample_input = next(iter(train_loader))[0][:1].to(self.device)
        macs, profile_params = get_model_profile(self.model, sample_input)
        before_train_inference_time = measure_inference_time(self.model, sample_input)

        epoch_cost_times = []
        max_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated(self.device)
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time_start = time.time()
            loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{self.args.train_epochs}]", leave=False)
            for i, (batch_x,batch_y) in enumerate(loop):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 28==0:
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    iter_count = 0
                    time_now = time.time()
                    loop.set_postfix(
                        loss=loss.item(),
                        exp_left_time='{:.1f}min'.format(left_time / 60)
                    )
                
                loss.backward()
                model_optim.step()

            cost_time=time.time() - epoch_time_start
            epoch_cost_times.append(cost_time)
            if cost_time < 60:
                print("Epoch: {} cost time: {:.1f}s".format(epoch+1, cost_time))
            elif cost_time < 3600:
                print("Epoch: {} cost time: {:.1f}min".format(epoch + 1, cost_time / 60))
            else:
                print("Epoch: {} cost time: {:.1f}h".format(epoch + 1, cost_time / 3600))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Test Loss: {4:.3f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
        #in_next_epoch
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated(self.device)
            max_memory = max_memory / 1024 ** 3 # GB
        avg_epoch_time = sum(epoch_cost_times[2:]) / len(epoch_cost_times[2:])

        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')

        after_train_inference_time = measure_inference_time(self.model, sample_input)

        # efficiency result save
        folder_path = './results/log/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        efficiency_log = os.path.join(folder_path+"eff.txt")
        with open(efficiency_log, 'a') as f:
            f.write(f"Custom_Model_Parameters:\t{custom_params}\n")
            f.write(f"Profile Model Parameters:\t{profile_params}\n")
            f.write(f"MACs:\t{macs}\n")
            f.write(f"Before Train Inference Time (ms/Inference):\t{before_train_inference_time:.3f}\n")
            f.write(f"After Train Inference Time (ms/Inference):\t{after_train_inference_time:.3f}\n")
            f.write(f"Max Memory Usage (GB):\t{max_memory:.3f}\n")
            f.write(f"Training Time(s/epoch):\t{avg_epoch_time:.3f}\n")
        with open('./results/eff.txt', 'a') as f:
            f.write(f"\n{setting}\t\t")
            f.write(f"Custom_Model_Parameters:\t{custom_params}\t")
            f.write(f"Profile Model Parameters:\t{profile_params}\t")
            f.write(f"MACs:\t{macs}\t")
            f.write(f"Before Train Inference Time (ms/Inference):\t{before_train_inference_time:.3f}\t")
            f.write(f"After Train Inference Time (ms/Inference):\t{after_train_inference_time:.3f}\t")
            f.write(f"Max Memory Usage (GB):\t{max_memory:.3f}\t")
            f.write(f"Training Time(s/epoch):\t{avg_epoch_time:.3f}")
        return self.model

    def test(self, setting, save_pred = False, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()

        inputs = []
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(test_data, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
                    inputs.append(batch_x.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/log/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{:.3f}, mae:{:.3f}'.format(mse, mae))
        with open('./results/result.txt', 'a') as file:
            tmptxt = "{}	mse:	{:.3f}	, mae:	{:.3f}".format(setting, mse, mae)
            file.write(tmptxt + '\n')

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            inputs = np.concatenate(inputs, axis=0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
            np.save(folder_path+'input.npy', inputs)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse = False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y
    
    def eval(self, setting, save_pred = False, inverse = False):
        #evaluate a saved model
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
            scale = True,
            scale_statistic = args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        self.model.eval()

        inputs = []
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
                    inputs.append(batch_x.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{:.3f}, mae:{:.3f}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            inputs = np.concatenate(inputs, axis=0)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'input.npy', inputs)
        return mae, mse, rmse, mape, mspe
