import numpy as np
import os
import drawSvg as draw

from .icon_base import IconBase

class IconBilliards(IconBase):
    def __init__(self) -> None:
        super().__init__()

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

        d = self.pose_postprocess(d, pose)

        return d
    
    def pose_postprocess(self, d, pose):
        return d