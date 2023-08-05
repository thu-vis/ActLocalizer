import os
import numpy as np
import torch
import numpy as np
from utils.model import SFNET
from utils.dataset import Dataset
from utils.dataset.video_dataset import np_load_old
from utils.dataset.config import config
import options
import cv2

# np_load_old = np.load
# np_save_old = np.save

# # modify the default parameters of np.load
# np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
# np.save = lambda *a, **k: np_save_old(*a, allow_pickle=True, **k)


feature_root = "/data/haoze/VideoModel/Thumos14_SwinB/"

videoname_path = "/data/changjian/VideoModel/SF-Net/data/Thumos19-Annotations/videoname.npy"

features = []

videonames = np.load(videoname_path)
total_count = 0
for videoname in videonames:
    videoname = videoname.decode()
    filename = feature_root + videoname + ".npy"    
    if os.path.exists(filename):
        total_count += 1
        feat = np_load_old(filename)
        feat = np.concatenate([feat, feat], axis=1)
        features.append(feat)
        # import IPython; IPython.embed(); exit()
        video_path = os.path.join("/data/changjian/VideoModel/Data/Thumos14/videos", videoname + ".mp4")
        cap=cv2.VideoCapture(video_path)
        frame_num = cap.get(7)
        assert frame_num // 16 == feat.shape[0]
    else:
        features.append([[0]])

np.save("/data/changjian/VideoModel/SF-Net/data/ThumosSwin-Features.npy", features)

import IPython; IPython.embed(); exit()

