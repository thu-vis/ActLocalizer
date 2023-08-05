import numpy as np
import os

from .icon_base import IconBase


# 0: head
# 1,2: 'left_shoulder', 'right_shoulder'
# 3,4: 'left_elbow', 'right_elbow',
# 5,6: 'left_wrist', 'right_wrist',
# 7,8: 'left_hip', 'right_hip',
# 9,10: 'left_knee', 'right_knee',
# 11,12:'left_ankle', 'right_ankle'

class IconWeightlifting(IconBase):
    def __init__(self) -> None:
        super().__init__()

    def pose_preprocess(self, pose, center):
        left_delta_x = pose[1][0] * 0.5
        right_delta_x = pose[2][0] * 0.5
        pose[[1,3,5,7,9,11], 0] += left_delta_x
        pose[[2,4,6,8,10,12], 0] += right_delta_x
    
    def pose_postprocess(self, d, pose):
        left_wrist = pose[5]
        right_wrist = pose[6]
        middle = (left_wrist + right_wrist) / 2
        left = middle.copy()
        left[0] = (left_wrist[0] - middle[0]) * 2.5 + middle[0]
        right = middle.copy()
        right[0] = (right_wrist[0] - middle[0]) * 2.5 + middle[0]

        self.draw_lines(3, left, right, d, False, False)

        left_small_up = [left[0] + 5, left[1] + 6]
        left_small_bottom = [left[0] + 5, left[1] - 6]
        self.draw_lines(3, left_small_up, left_small_bottom, d, False, False)
        left_large_up = [left[0], left[1] + 9]
        left_large_bottom = [left[0], left[1] - 9]
        self.draw_lines(3, left_large_up, left_large_bottom, d, False, False)

        right_small_up = [right[0] - 5, left[1] + 6]
        right_small_bottom = [right[0] - 5, left[1] - 6]
        self.draw_lines(3, right_small_up, right_small_bottom, d, False, False)
        right_large_up = [right[0], left[1] + 9]
        right_large_bottom = [right[0], left[1] - 9]
        self.draw_lines(3, right_large_up, right_large_bottom, d, False, False)

        return d