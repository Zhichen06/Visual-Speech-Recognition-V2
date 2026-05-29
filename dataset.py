from torch.utils.data import Dataset
import cv2
import os
import glob
import numpy as np
import random
from .cvtransforms import *
import torch
from collections import defaultdict
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE


jpeg = TurboJPEG()
class LRW1000_Dataset(Dataset):
    
    def __init__(self, phase, args):
        
        self.args = args
        self.data = []
        self.phase = phase
        
        #【改动】：数据路径
        base_dir = '/share/home/dwj/learn-an-effective-lip-reading-model-without-pains/scripts/LRW1000_Public_pkl_jpeg'
        
        if(self.phase == 'train'):
            self.index_root = os.path.join(base_dir, 'trn')
        else:
            self.index_root = os.path.join(base_dir, 'tst')                        
        
        self.data = glob.glob(os.path.join(self.index_root, '*.pkl'))
        

                                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pkl = torch.load(self.data[idx])
        video = pkl.get('video')
        video = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in video]        
        video = np.stack(video, 0)
        video = video[:,:,:,0]
        
        
        if(self.phase == 'train'):
            video = RandomCrop(video, (88, 88))
            video = HorizontalFlip(video)
        elif self.phase == 'val' or self.phase == 'test':
            video = CenterCrop(video, (88, 88))      
        
        pkl['video'] = torch.FloatTensor(video)[:,None,...] / 255.0        
                
        return pkl
