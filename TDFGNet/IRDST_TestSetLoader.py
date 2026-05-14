"""
Datasets dataloader for inference.
"""
import math
import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
import glob
import logging


import transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IRDST_TestDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=7, test_size=256):
        self.num_frames = num_frames
        self.test_size = test_size

        self.test_seqs_file = '/home/tcs1/data2/lyc/DTUM_main/dataset/IRDST/IRDST/test.txt'
        self.img_path = '/home/tcs1/data2/lyc/DTUM_main/dataset/IRDST/IRDST/images'
        self.mask_path = '/home/tcs1/data2/lyc/DTUM_main/dataset/IRDST/IRDST/masks'

        self.frames_info = {'dataset': {}}
        self.img_ids = []

        with open(self.test_seqs_file, 'r') as f:
            img_paths = [line.strip() for line in f if line.strip()]
            self.img_ids = []
            for path in img_paths:
                # 假设 mask 的路径和图片路径对应，只是根目录不同
                mask_path = path.replace('images', 'masks').replace('.png', '.png')
                # video_name 可以取文件夹名
                video_name = os.path.basename(os.path.dirname(path))
                frame_index = os.path.basename(path)[:-4]  # 去掉 .png
                self.img_ids.append(('dataset', video_name, frame_index))
                self.frames_info['dataset'].setdefault(video_name, []).append(frame_index)

        # 灰度图归一化参数
        self.mean = 0.485
        self.std = 0.229

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        import re

        dataset, video_name, frame_index = self.img_ids[idx]

        # 如果 frame_index 是字符串，需要解析成整数
        if isinstance(frame_index, str):
            match = re.search(r'\((\d+)\)', frame_index)  # 匹配括号内的数字
            if match:
                frame_index = int(match.group(1))
            else:
                digits = re.findall(r'\d+', frame_index)
                frame_index = int(digits[-1]) if digits else 0

        # 视频帧总数
        vid_len = len(self.frames_info[dataset][video_name])
        if vid_len == 0:
            raise ValueError(f"No frames found for video {video_name}")

        # 防止 frame_index 超出范围
        frame_index = min(max(frame_index, 0), vid_len - 1)

        # 计算多帧索引（循环取模，保证不会越界）
        half_frames = self.num_frames // 2
        frame_indices = [(x + vid_len) % vid_len for x in range(frame_index - half_frames,
                                                            frame_index + (self.num_frames + 1)//2)]

        imgs = []

        for fid in frame_indices:
            # 获取帧名
            frame_name = self.frames_info[dataset][video_name][fid]
            img_path = os.path.join(self.img_path, video_name, frame_name + '.png')

            # 读取灰度图
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Cannot read image: {img_path}")

            img = cv2.resize(img, (self.test_size, self.test_size))
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            imgs.append(np.expand_dims(img, axis=0))  # [1,H,W]

        # 多帧沿通道堆叠
        imgs = np.concatenate(imgs, axis=0)  # [num_frames, H, W]
        imgs = torch.from_numpy(imgs)  

        # 当前帧 mask 作为目标
        center_frame_name = self.frames_info[dataset][video_name][frame_index]
        mask_path = os.path.join(self.mask_path, video_name, center_frame_name + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        mask = cv2.resize(mask, (self.test_size, self.test_size), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 1
        mask = torch.from_numpy(np.expand_dims(mask.astype(np.float32), 0))

        m, n = mask.shape[1], mask.shape[2]

        return imgs, mask, m, n

