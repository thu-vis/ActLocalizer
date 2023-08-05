import numpy as np
import os

from .icon_base import IconBase

class IconHighJump(IconBase):
    def __init__(self) -> None:
        super().__init__()

    def pose_postprocess(self, d, pose):
        return d