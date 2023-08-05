from math import floor
import pickle
import numpy as np
import os
import cv2
from time import sleep, time

from SF_Net.utils.utils import check_dir

from ..utils.helper_utils import pickle_load_data, pickle_save_data
from ..utils.log_utils import logger
from PIL import Image

PROC_FLAG = {}

class VideoHelper(object):
    def __init__(self, vid, vname, vbuffer, vpose, fps, stride):
        self.vname = vname
        self.vbuffer = vbuffer
        self.vpose = vpose
        self.fps = fps
        self.stride = stride
        self.vid = vid
        self.read()
        self.is_ready = True

    def read(self):
        if os.path.exists(os.path.join(self.vbuffer, "finish.pkl")):
            logger.info("use video buffer for {}".format(self.vname))
            # self.frames = pickle_load_data(self.vbuffer)
            return
        global PROC_FLAG
        flag = PROC_FLAG.get(self.vid, 0)
        if flag == 0:
            PROC_FLAG[self.vid] = 1
            self._read()
            PROC_FLAG[self.vid] = 0
        else:
            while flag == 1 and not os.path.exists(self.vbuffer):
                sleep(2)
                flag = PROC_FLAG.get(self.vid, 0)


    def _read(self):
        t0 = time()
        self.cap = cv2.VideoCapture(self.vname)
        total_frames = self.cap.get(7)
        original_fps = self.cap.get(5)
        self.frames = {}
        fid = 0
        while True:
            t = fid * self.stride + self.stride / 2
            t = t / self.fps
            frame_idx = int(t * original_fps)
            if frame_idx >= total_frames:
                break
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            rval, frame = self.cap.read()
            self.frames[fid] = frame
            fid += 1
        
        # pickle_save_data("361_.pkl", self.frames)
        # return
        # self.frames = pickle_load_data("361.pkl")

        check_dir(self.vbuffer)
        for fid in range(len(self.frames)):
            img = self.frames[fid]
            img = img[:, :, ::-1]
            pil_img = Image.fromarray(img)
            # crop frame
            pil_img = self.clear_frame_black_bound(pil_img)
            # pil_img = self.crop_frame_in_ratio(pil_img, 0.7, mode="lt")
            pil_img = self.crop_frame_in_ratio(pil_img, 0.9) 
            pil_img.save(os.path.join(self.vbuffer, 
                str(self.vid) + "-" + str(fid) + ".jpg"))
        pickle_save_data(os.path.join(self.vbuffer, "finish.pkl"), ["finish"])
        logger.info("save buffer time: {}s".format(time() - t0))
        return 

    # clear frame black bound
    def clear_frame_black_bound(self, img: Image, maxcrop=60, rate = 0.9):
        # from IPython import embed
        # embed()
        xstart, ystart = 0, 0
        xsize, ysize = img.size
        xend, yend = xsize, ysize
        assert img.mode == "RGB"

        pix = img.load()
        for _ in range(maxcrop):
            x = xstart
            color = 0
            for y in range(ysize):
                if pix[x, y] != (0,0,0):
                    color += 1
            if 1 - color / ysize <= rate:
                break
            xstart = xstart + 1
        
        for _ in range(maxcrop):
            x = xend - 1
            color = 0
            for y in range(ysize):
                if pix[x, y] != (0,0,0):
                    color += 1
            if 1 - color / ysize <= rate:
                break
            xend = xend - 1

        for _ in range(maxcrop):
            y = ystart
            color = 0
            for x in range(xsize):
                if pix[x, y] != (0,0,0):
                    color += 1
            if 1 - color / xsize <= rate:
                break
            ystart = ystart + 1
        
        for _ in range(maxcrop):
            y = yend - 1
            color = 0
            for x in range(xsize):
                if pix[x, y] != (0,0,0):
                    color += 1
            if 1 - color / xsize <= rate:
                break
            yend = yend - 1

        img = img.crop((xstart, ystart, xend, yend))
        return img

    # find pose and crop frame by pose
    def find_pose(self, poses, poseid):
        if len(poses[poseid]) > 0:
            return poses[poseid]
        pmax = len(poses) - 1
        i, j = 1, 1
        while i < 20:
            if poseid - i < 0:
                i = 100
                break
            if len(poses[poseid - i]) > 0:
                break
            i += 1
        while j < 20:
            if poseid + j > pmax:
                j = 100
                break
            if len(poses[poseid + j]) > 0:
                break
            j += 1

        if i > 19 and j > 19:
            return []
        if i < j:
            return poses[poseid - i]
        else:
            return poses[poseid + j]

    def find_max_pose(self, pose):
        grid = 25
        num = int(len(pose) / (2 * grid))
        if num < 1:
            return pose, [-1,-1]
        ln = num * grid
        all_pts = np.array(pose).reshape((ln, 2))
        S = []
        for i in range(num):
            pts = all_pts[i * grid: i * grid + 16]
            xmin, ymin = np.min(pts, axis=0)
            xmax, ymax = np.max(pts, axis=0)
            S.append((xmax - xmin) * (ymax - ymin))
        maxi = np.array(S).argmax()
        pts = all_pts[maxi * grid: maxi * grid + 16]
        xctr, yctr = np.mean(pts, axis=0)
        pose = pts.flatten().tolist()
        return pose, [xctr, yctr]

    def crop_frame_by_pose(self, img: Image, pose: list) -> Image:
        ln = int(len(pose) / 2)
        pts = np.array(pose).reshape((ln, 2))
        valid_pt_ids = list(np.where(pts.any(axis=1))[0])
        if len(valid_pt_ids) == 0:
            size = max(img.size)
            img = img.resize([size, size])
            return img
        pts = pts[valid_pt_ids]
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)
        xsize, ysize = img.size
        if xmin < 0:
            xmin = 0
        if xmax >= xsize:
            xmax = xsize - 1 
        if ymin < 0:
            ymin = 0
        if ymax >= ysize:
            ymax = ysize - 1 
        width = max(xmax - xmin + 1, ymax - ymin + 1) + 10 

        # judge x
        if width > xsize:
            xstart, xend = 0, xsize
        else:
            delta = width - xmax + xmin
            delta1 = round(delta / 2)
            delta2 = delta - delta1
            xstart, xend = 0, 0
            if xmin - delta1 >= 0 and xmax + delta2 <= xsize:
                xstart, xend = xmin - delta1, xmax + delta2
            elif xmin - delta2 >= 0 and xmax + delta1 <= xsize:
                xstart, xend = xmin - delta2, xmax + delta1
            elif xmin < delta1:
                xstart, xend = 0, width
            else:
                xstart, xend = xsize - width, xsize

        # judge y
        if width > ysize:
            ystart, yend = 0, ysize
        else:
            delta = width - ymax + ymin
            delta1 = round(delta / 2)
            delta2 = delta - delta1
            ystart, yend = 0, 0
            if ymin - delta1 >= 0 and ymax + delta2 <= ysize:
                ystart, yend = ymin - delta1, ymax + delta2
            elif ymin - delta2 >= 0 and ymax + delta1 <= ysize:
                ystart, yend = ymin - delta2, ymax + delta1
            elif ymin < delta1:
                ystart, yend = 0, width
            else:
                ystart, yend = ysize - width, ysize

        img = img.crop((xstart, ystart, xend, yend))
        size = max(img.size)
        img = img.resize([size, size])
        return img
    
    # crop frame in ratio
    def crop_frame_in_ratio(self, img: Image, ratio=0.7, mode="ctr") -> Image:
        # mode: ctr/lt/ld/rt/rd
        xsize, ysize = img.size 
        rxsize, rysize = round(xsize * ratio), round(ysize * ratio)
        deltax, deltay = 0, 0
        if mode == "ctr":
            deltax, deltay = round((xsize - rxsize) / 2), round((ysize - rysize) / 2)
        if mode == "ld" or mode == "rd":
            deltay = ysize - rysize
        if mode == "rt" or mode == "rd":
            deltax = xsize - rxsize
        img = img.crop((deltax, deltay, deltax + rxsize, deltay + rysize))
        size = max(img.size)
        img = img.resize([size, size])
        return img

    def get_single_frame(self, fid):
        while True:
            if hasattr(self, "is_ready") and self.is_ready:
                break
            sleep(2)
        return os.path.join(self.vbuffer, str(self.vid) + "-" + str(fid) + ".jpg")
    
    # def get_single_frame(self, fid):
    #     t = fid * self.stride + self.stride / 2
    #     t = t / self.fps
    #     original_fps = self.cap.get(5) # 5 stands for CV_CAP_PROP_FPS 
    #     frame_idx = int(t * original_fps)
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    #     rval, frame = self.cap.read()
    #     return frame

    