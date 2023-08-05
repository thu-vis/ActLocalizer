'''
Author: Changjian Chen
Date: 2021-11-09 16:28:56
LastEditTime: 2021-11-09 18:44:52
LastEditors: Changjian Chen
Description: 
FilePath: /VideoVis/SF-Net/utils/dataset/video_reader.py

'''
import os
from .config import config
import cv2
from PIL import Image

def read_frames(videoname, dataname, frame_id, stride, fps):
    st = int(frame_id) * stride / fps
    et = (int(frame_id) * stride + stride - 1) / fps
    videopath = os.path.join(config.raw_data_root, dataname, "videos", videoname + ".mp4")
    # print(videopath)
    video_fps = config.data_config[dataname]["video_fps"]
    st_frame = int(st * video_fps)
    et_frame = int(et * video_fps)
    # print(st_frame, et_frame)
    cap = cv2.VideoCapture(videopath)
    frames = []
    for i in range(st_frame, et_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames

def save_frames(dir, frames):
    for idx, frame in enumerate(frames):
        im = Image.fromarray(frame)
        im.save(os.path.join(dir, str(idx) + ".jpg"))
    return True