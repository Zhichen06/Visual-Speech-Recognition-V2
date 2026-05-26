# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import time
from model import *
import torch.optim as optim 
import random
from LSR import LSR
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

# ==================== 日志拦截器 ====================
class DualStreamInterceptor(object):
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
# ====================================================

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser.add_argument('--gpus', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--n_class', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--max_epoch', type=int, required=True)

parser.add_argument('--test', type=str2bool, required=False, default=False)
parser.add_argument('--weights', type=str, required=False, default=None)
parser.add_argument('--save_prefix', type=str, required=True)
parser.add_argument('--dataset', type=str, required=False, default='lrw1000')
parser.add_argument('--border', type=str2bool, required=False, default=False)
parser.add_argument('--mixup', type=str2bool, required=False, default=True)
parser.add_argument('--label_smooth', type=str2bool, required=False, default=True)
parser.add_argument('--se', type=str2bool, required=False, default=True)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# [修改点]: 日志文件将保存在专门的 logs 文件夹内，方便论文素材归档
if not args.test:
    log_dir = os.path.join(args.save_prefix, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    _ACADEMIC_LOG_FILE = os.path.join(log_dir, f"train_{log_time}.log")
    sys.stdout = DualStreamInterceptor(_ACADEMIC_LOG_FILE)
    print(f"Log stream system activated. Target path: {_ACADEMIC_LOG_FILE}")

# [修改点]: 移除了 args.border = False 和 args.dataset_dir 的全局硬编码

if(args.dataset == 'lrw'):
    from utils import LRWDataset as Dataset
elif(args.dataset == 'lrw1000'):    
    from utils.dataset_lrw1000 import LRW1000_Dataset as Dataset
else:
    raise Exception('lrw or lrw1000')    

video_model = VideoModel(args).cuda()

def parallel_model(model):
    model = nn.DataParallel(model)
    return model        

def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
optim_video = optim.Adam(video_model.parameters(), lr = lr, weight_decay=1e-4)     
scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max = args.max_epoch, eta_min=5e-6)

if(args.weights is not None):
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))    
    load_missing(video_model, weight.get('video_model'))
          
video_model = parallel_model(video_model)

def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
            batch_size = batch_size, 
            num_workers = num_workers,   
            shuffle = shuffle,         
            drop_last = False,
            pin_memory=True)
    return loader

def add_msg(msg, k, v):
    if(msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg    

def test(epoch=None):
    with torch.no_grad():
        dataset = Dataset('val', args)
        print(f'\nStart Testing (Epoch {epoch}), Data Length: {len(dataset)}')
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)        
        
        v_acc = []
        v_acc_top5 = [] 
        total = 0

        for (i_iter, input) in enumerate(loader):
            video_model.eval()
            
            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            total = total + video.size(0)
            border = input.get('duration').cuda(non_blocking=True).float()
            
            with autocast():
                if(args.border):
                    y_v = video_model(video, border)                                            
                else:
                    y_v = video_model(video)                                            
            
            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            
            _, top5_preds = y_v.topk(5, dim=-1)
            top5_correct = top5_preds.eq(label.view(-1, 1).expand_as(top5_preds))
            v_acc_top5.extend(top5_correct.sum(dim=-1).clamp(max=1).cpu().numpy().tolist())
            
            toc = time.time()
            if(i_iter % 10 == 0):  
                msg = ''              
                msg = add_msg(msg, 'v_acc_top1={:.5f}', np.array(v_acc).reshape(-1).mean())                
                msg = add_msg(msg, 'v_acc_top5={:.5f}', np.array(v_acc_top5).reshape(-1).mean())                
                msg = add_msg(msg, 'eta={:.5f}', (toc-tic)*(len(loader)-i_iter)/3600.0)
                print(msg)            

        acc = float(np.array(v_acc).reshape(-1).mean())
        acc_top5 = float(np.array(v_acc_top5).reshape(-1).mean())
        
        exam_report = f"""
==================================================
        [Epoch {epoch} Final Exam Results]
        Top-1 Accuracy : {acc:.5f}
        Top-5 Accuracy : {acc_top5:.5f}
==================================================
        """
        print(exam_report)
        
        msg = 'v_acc_top1_{:.5f}_top5_{:.5f}_'.format(acc, acc_top5)
        return acc, msg                                    

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)

def train():            
    dataset = Dataset('train', args)
    print('Start Training, Data Length:',len(dataset))
    
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)
        
    max_epoch = args.max_epoch    
    ce = nn.CrossEntropyLoss()

    tot_iter = 0
    best_acc = 0.0
    alpha = 0.2
    scaler = GradScaler()             
    
    for epoch in range(max_epoch):
        for (i_iter, input) in enumerate(loader):
            tic = time.time()           
            
            video_model.train()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True).long()     
            border = input.get('duration').cuda(non_blocking=True).float()
            
            loss = {}
            
            if(args.label_smooth):
                loss_fn = LSR()
            else:
                loss_fn = nn.CrossEntropyLoss()
            
            with autocast():
                if(args.mixup):
                    video_gpu = video.contiguous()
                    
                    lambda_ = np.random.beta(alpha, alpha)
                    index = torch.randperm(video_gpu.size(0)).cuda(non_blocking=True)
                    
                    mix_video = lambda_ * video_gpu + (1 - lambda_) * video_gpu[index, :]
                    label_a, label_b = label, label[index]            
                    
                    if(args.border):
                        y_v = video_model(mix_video, border)
                    else:
                        y_v = video_model(mix_video)
                        
                    loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)
                else:
                    if(args.border):
                        y_v = video_model(video, border)
                    else:
                        y_v = video_model(video)
                    loss_bp = loss_fn(y_v, label)
                                    
            loss['CE V'] = loss_bp
                
            optim_video.zero_grad()   
            scaler.scale(loss_bp).backward()  
            scaler.step(optim_video)
            scaler.update()
            
            toc = time.time()
            time_str = datetime.now().strftime('%H:%M:%S')
            current_batch_size = video.size(0)
            
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            else:
                gpu_mem_allocated = 0.0
                gpu_mem_reserved = 0.0
            
            msg = 'time={},epoch={},step={:04d},train_iter={},eta={:.5f}'.format(time_str, epoch, i_iter, tot_iter, (toc-tic)*(len(loader)-i_iter)/3600.0)
            for k, v in loss.items():                                                
                msg += ',{}={:.5f}'.format(k, v)
            msg = msg + str(',lr=' + str(showLR(optim_video)))                    
            msg = msg + str(',best_acc={:2f}'.format(best_acc))
            msg = msg + f',batch_size={current_batch_size},mixup={args.mixup},gpu_alloc={gpu_mem_allocated:.2f}GB,gpu_reserv={gpu_mem_reserved:.2f}GB'
            print(msg)                                
            
            if(i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0)):
                acc, msg_test = test(epoch=epoch)

                if(acc > best_acc):
                    savename = '{}_iter_{}_epoch_{}_{}.pt'.format(args.save_prefix, tot_iter, epoch, msg_test)
                    temp = os.path.split(savename)[0]
                    if(not os.path.exists(temp)):
                        os.makedirs(temp)                    
                    torch.save({'video_model': video_model.module.state_dict()}, savename)         
                     
                if(tot_iter != 0):
                    best_acc = max(acc, best_acc)    
                    
            tot_iter += 1        
            
        scheduler.step()            
        
if(__name__ == '__main__'):
    if(args.test):
        acc, msg = test(epoch="Test Mode")
        print(f'Final Test Acc = {acc}')
        exit()
    train()
