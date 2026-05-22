# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
from turbojpeg import TurboJPEG, TJPF_GRAY
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_single_file(args):
    src_path, dst_path = args
    if os.path.exists(dst_path):
        return True
    try:
        jpeg = TurboJPEG()
        pkl_data = torch.load(src_path, map_location='cpu')
        video_bytes = pkl_data.get('video')
        label = pkl_data.get('label')
        if video_bytes is None:
            return False
        decoded_video = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in video_bytes]
        video_np = np.array(decoded_video, dtype=np.uint8)
        processed_data = {
            'video': video_np,
            'label': label,
            'is_decoded': True
        }
        torch.save(processed_data, dst_path)
        return True
    except Exception as e:
        print(f"\nError processing file {src_path}: {str(e)}", file=sys.stderr)
        return False

def file_generator(src_base, dst_base, sub_dirs):
    for sub in sub_dirs:
        src_dir = os.path.join(src_base, sub)
        dst_dir = os.path.join(dst_base, sub)
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(src_dir):
            continue
        print(f"Scanning subdirectory: {sub} ...")
        with os.scandir(src_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.pkl'):
                    yield (entry.path, os.path.join(dst_dir, entry.name))

def main():
    SRC_BASE = "/lip"               
    DST_BASE = "/lip/decoded_dataset"   
    SUB_DIRS = ["trn", "val", "tst"] 
    
    num_workers = min(12, max(4, cpu_count() - 4)) 
    print(f"CPU cores: {cpu_count()} | Allocated processes: {num_workers}")
    print(f"Source: {SRC_BASE}")
    print(f"Target: {DST_BASE}")
    
    pool = Pool(processes=num_workers)
    tasks = file_generator(SRC_BASE, DST_BASE, SUB_DIRS)
    
    print("Starting multiprocessing pipeline...")
    success_count = 0
    
    for result in tqdm(pool.imap_unordered(process_single_file, tasks, chunksize=64)):
        if result:
            success_count += 1

    pool.close()
    pool.join()
    print(f"\nFinished. Total decoded files: {success_count}")

if __name__ == '__main__':
    main()
