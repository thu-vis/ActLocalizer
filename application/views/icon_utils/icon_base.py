from ipaddress import ip_address
from tracemalloc import get_object_traceback
import numpy as np
import os
import drawSvg as draw
from PIL import Image
import io

from ..utils.helper_utils import pickle_load_data

def intersect(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]
    min_xy_a = box_a[:, :2][:, np.newaxis, :].repeat(axis=1, repeats=B)
    min_xy_b = box_b[:, :2][np.newaxis, :, :].repeat(axis=0, repeats=A)
    max_xy_a = box_a[:, 2:4][:, np.newaxis, :].repeat(axis=1, repeats=B)
    max_xy_b = box_b[:, 2:4][np.newaxis, :, :].repeat(axis=0, repeats=A)
    max_xy = np.minimum(max_xy_a, max_xy_b)
    min_xy = np.maximum(min_xy_a, min_xy_b)
    inter = (max_xy - min_xy).clip(0)
    areas = inter[:, :, 0] * inter[:, :, 1]
    return areas

def jacard(box_pred, box_truth):
    A = box_pred.shape[0]
    B = box_truth.shape[0]
    if B == 0:
        return np.zeros((A, 1))
    inter = intersect(box_pred, box_truth)
    area_a = ((box_pred[:, 2]-box_pred[:, 0]) *
              (box_pred[:, 3]-box_pred[:, 1]))[:, np.newaxis].repeat(axis=1, repeats=B)
    area_b = ((box_truth[:, 2]-box_truth[:, 0]) *
              (box_truth[:, 3]-box_truth[:, 1]))[np.newaxis, :].repeat(axis=0, repeats=A)
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def get_bbox(pose):
    min_pose = pose.copy()
    min_pose[min_pose == 0] = 1000
    x_min = min_pose[:, :, 0].min(axis=1).reshape(-1,1)
    y_min = min_pose[:, :, 1].min(axis=1).reshape(-1,1)
    x_max = pose[:, :, 0].max(axis=1).reshape(-1,1)
    y_max = pose[:, :, 1].max(axis=1).reshape(-1,1)
    res = np.concatenate([x_min, y_min, x_max, y_max], axis=1)
    return res

def get_area(pose):
    pose = get_bbox(pose)
    area = (pose[:, 2]-pose[:, 0]) * (pose[:, 3]-pose[:, 1])
    return area

def save_square(d, filename):
    a = d.rasterize()
    img = Image.open(io.BytesIO(a.pngData))
    data = np.array(img)
    opacity = data[:, :, 3]
    x, y = opacity.nonzero()
    x_min, x_max = x.min() - 2, x.max() + 2
    y_min, y_max = y.min() - 2, y.max() + 2
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    width = x_max - x_min
    if width % 2 == 1:
        width += 1
    height = y_max - y_min
    if height % 2 == 1:
        height += 1
    if width > height:
        x_left = x_min
        x_right = x_max
        y_top = center_y - width // 2
        y_bottom = center_y + width // 2
    else:
        x_left = center_x - height // 2
        x_right = center_x + height // 2
        y_top = y_min 
        y_bottom = y_max 
    data = data[x_left:x_right+1, y_top:y_bottom+1, :]
    img = Image.fromarray(data)
    img.save(filename)


class IconBase(object):
    def __init__(self) -> None:
        None

    def init(self, openpose_filepath, baidupose_filepath):
        self.openpose_filepath = openpose_filepath
        self.baidupose_filepath = baidupose_filepath

        self.openpose = pickle_load_data(self.openpose_filepath)
        if os.path.exists(self.baidupose_filepath):
            self.baidupose = pickle_load_data(self.baidupose_filepath)
        else:
            self.baidupose = []
        self.ensemble_data()
        self.pickle_the_one()

    def ensemble_data(self):
        self.all_posedata = []
        if len(self.baidupose) == 0:
            for i in range(len(self.openpose)):
                self.all_posedata.push(np.array([]).reshape([-1, 17, 2]))
        assert len(self.openpose) == len(self.baidupose)
        for i in range(len(self.openpose)):
            self.openpose[i] = np.array(self.openpose[i]).reshape(-1, 25, 2)
            self.baidupose[i] = np.array(self.baidupose[i]).reshape(-1, 17, 2)
            op = get_bbox(self.openpose[i])
            bp = get_bbox(self.baidupose[i])
            iou = jacard(bp, op)
            iou = iou.max(axis=1)
            self.all_posedata.append(self.baidupose[i][iou>0.3])

    def pickle_the_one(self):
        self.posedata = []
        for i in range(len(self.all_posedata)):
            if len(self.all_posedata[i]) == 0:
                self.posedata.append([])
            else:
                pp = self.all_posedata[i]
                area = get_area(pp)
                max_idx = area.argmax()
                self.posedata.append(self.all_posedata[i][max_idx])


    def get_pose_by_id(self, id):
        # print("len:", len(self.pose_data[id]) / 34)
        # poses = self.pose_data[id][:34]
        # poses = self.pose_data[id][34: 34 * 2]
        # poses = self.pose_data[id][34 * 2: 34 * 3]
        try:
            pose = self.posedata[id]
            pose = self.trans_pose(pose)
            return pose
        except:
            return None

    # 0: head
    # 1,2: 'left_shoulder', 'right_shoulder'
    # 3,4: 'left_elbow', 'right_elbow',
    # 5,6: 'left_wrist', 'right_wrist',
    # 7,8: 'left_hip', 'right_hip',
    # 9,10: 'left_knee', 'right_knee',
    # 11,12:'left_ankle', 'right_ankle'
    def trans_pose(self, pose):
        # assert len(pose) == 34
        # pose = np.array(pose).reshape(17, 2)
        if len(pose) == 0:
            return None
        head = pose[:5, :].mean(axis=0)
        pose = np.concatenate([head.reshape(1, 2), pose[5:, :]], axis=0)
        return pose

    def _draw_lines(self, width, start, end, stroke, round):
        if round:
            r = draw.Path(stroke_width=width, stroke=stroke, 
                fill="none", stroke_linejoin="round")
            r.M(start[0], start[1])
            r.L(end[0], end[1])
            r.L(start[0], start[1])
            r.L(end[0], end[1])
        else:
            r = draw.Path(stroke_width=width, stroke=stroke, 
                fill="none")
            r.M(start[0], start[1])
            r.L(end[0], end[1])
        return r

    def draw_lines(self, width, start, end, d, white_stroke, round=True):
        if white_stroke:
            r = self._draw_lines(width+2, start, end, "white", round)
            d.append(r)

        r = self._draw_lines(width, start, end, "black", round)
        d.append(r)
    
    def pose_preprocess(self, pose, center):
        None

    def basic_drawing(self, pose):
        center = pose[[1,2, 7, 8], :].mean(axis=0)
        pose = pose - center

        self.pose_preprocess(pose, center)

        pose[:, 1] = - pose[:, 1]
        min_pos = pose.min(axis=0)
        max_pos = pose.max(axis=0)
        size = max_pos - min_pos
        size = max(size) * 10
        d = draw.Drawing(size, size, origin='center', displayInline=False)

        # head
        r = draw.Circle(cx=pose[0][0], cy=pose[0][1], r=7, fill="black")
        d.append(r)

        # body
        r = draw.Path(stroke_width=0, fill="black")
        r.M(pose[1][0], pose[1][1])
        r.L(pose[2][0], pose[2][1])
        r.L(pose[8][0], pose[8][1])
        r.L(pose[7][0], pose[7][1])
        d.append(r)

        self.draw_lines(6, pose[1], pose[2], d, False) # shoulder
        self.draw_lines(6, pose[3], pose[1], d, False) # left upper arm
        self.draw_lines(6, pose[2], pose[4], d, False) # right upper arm
        self.draw_lines(6, pose[5], pose[3], d, True) # left forearm
        self.draw_lines(6, pose[4], pose[6], d, True) # right forearm

        self.draw_lines(6, pose[7], pose[8], d, False) # hip
        self.draw_lines(6, pose[9], pose[7], d, False) # left lap
        self.draw_lines(6, pose[8], pose[10], d, False) # right lap
        self.draw_lines(6, pose[9], pose[11], d, True) # left leg
        self.draw_lines(6, pose[10], pose[12], d, True) # right leg

        d = self.pose_postprocess(d, pose)

        return d
    
    def pose_postprocess(self, d, pose):
        return d