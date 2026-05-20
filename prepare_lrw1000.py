# -*- coding: utf-8 -*-
import os
import sys
import cv2
import torch
import time  # 补上了这个关键的 import
import numpy as np
from torch.utils.data import Dataset, DataLoader
from turbojpeg import TurboJPEG

# 强制 CPU 模式，规避 Docker 环境下的 EGL 报错
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

# ================= HACK: MediaPipe Hidden Path =================
try:
    import mediapipe as mp
    mp_root = mp.__path__[0]
    hidden_path = os.path.join(mp_root, 'python')
    if hidden_path not in sys.path:
        sys.path.append(hidden_path)
    
    from solutions import face_mesh as mp_face_mesh
    print(">>> [Path Hack] MediaPipe Solutions loaded successfully")
except Exception as e:
    print(f">>> MediaPipe loading failed: {e}")
    sys.exit(1)

# ================= Configuration =================
DATA_ROOT = '/remote-home/images/lip_images'
ANNO_ALL = '/remote-home/project/info/all_audio_video.txt'
INFO_DIR = '/remote-home/project/info/'
OUTPUT_DIR = '/remote-home/LRW1000_Processed_PKL/'

LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 
                318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 
                269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]

jpeg = TurboJPEG()

class LRW1000_Dataset(Dataset):
    def __init__(self, index_file, target_dir, global_pinyins):
        self.data = []
        self.target_dir = target_dir
        self.pinyins = global_pinyins
        self.padding = 40
        self.face_mesh = None 

        if not os.path.exists(index_file):
            print(f"!!! Error: {index_file} missing")
            return

        with open(index_file, 'r', encoding='utf-8') as f:
            lines = [line.strip().split(',') for line in f.readlines()]
        
        for line in lines:
            if len(line) < 5: continue
            folder, pinyin, op, ed = line[0], line[2], line[3], line[4]
            try:
                op_f, ed_f = int(float(op)*25)+1, int(float(ed)*25)+1
                if pinyin in self.pinyins:
                    label = self.pinyins.index(pinyin)
                    self.data.append((folder, op_f, ed_f, label))
            except: continue
        print(f">>> Dataset initialized: {len(self.data)} samples.")

    def load_images(self, path, op, ed):
        if self.face_mesh is None:
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True, 
                max_num_faces=1,
                min_detection_confidence=0.5
            )

        center = (op + ed) // 2
        start_frame = center - self.padding // 2
        
        anchor_path = os.path.join(path, f'{center}.jpg')
        if not os.path.exists(anchor_path):
            files = sorted([f for f in os.listdir(path) if f.endswith('.jpg')])
            if not files: return None
            anchor_path = os.path.join(path, files[len(files)//2])

        img = cv2.imread(anchor_path)
        if img is None: return None
        h, w, _ = img.shape
        
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        box = [int(h*0.5), int(h*0.8), int(w*0.3), int(w*0.7)]
        
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            pts = np.array([(lms[i].x * w, lms[i].y * h) for i in LIPS_INDICES])
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            box = [int(min_y - 20), int(max_y + 20), int(min_x - 20), int(max_x + 20)]

        processed = []
        for i in range(start_frame, start_frame + self.padding):
            p = os.path.join(path, f'{i}.jpg')
            if os.path.exists(p):
                frame = cv2.imread(p)
                crop = frame[max(0,box[0]):min(h,box[1]), max(0,box[2]):min(w,box[3])]
                crop = cv2.resize(crop, (112, 112))
            else:
                crop = np.zeros((112, 112, 3), dtype=np.uint8)
            processed.append(jpeg.encode(crop))
        return processed

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        try:
            folder, op, ed, label = self.data[idx]
            inputs = self.load_images(os.path.join(DATA_ROOT, folder), op, ed)
            if inputs is None: return False
            
            # 修正了变量名不统一的问题
            save_path = os.path.join(self.target_dir, f"{folder}_{op}_{ed}.pkl")
            torch.save({'video': inputs, 'label': label}, save_path)
            return True
        except: return False

if __name__ == '__main__':
    print("Step 1: Building global dictionary...")
    with open(ANNO_ALL, 'r', encoding='utf-8') as f:
        pinyins = [l.strip().split(',')[3] for l in f.readlines() if len(l.split(',')) >= 4]
        global_pinyins = sorted(list(set(pinyins)))
    
    WORKERS = 16 

    for subset in ['trn', 'val', 'tst']:
        target = os.path.join(OUTPUT_DIR, subset)
        os.makedirs(target, exist_ok=True)
        
        ds = LRW1000_Dataset(os.path.join(INFO_DIR, f'{subset}_1000.txt'), target, global_pinyins)
        if len(ds) == 0: continue

        loader = DataLoader(ds, batch_size=1, num_workers=WORKERS, shuffle=False)
        
        print(f"\n>>> Processing [{subset}] with {WORKERS} workers...")
        start_t = time.time()
        
        for i, _ in enumerate(loader):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_t
                eta = (elapsed / i) * (len(ds) - i) / 3600
                print(f"[{subset}] Processed: {i}/{len(ds)} | ETA: {eta:.2f}h")