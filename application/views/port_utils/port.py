import numpy as np
import os

from ..case_utils import get_case_util
from ..model_utils import VideoModel
from ..utils.config_utils import config
from ..utils.log_utils import logger
from ..utils.helper_utils import *

from .port_helper import *

class Port(object):
    def __init__(self, dataname=None, step=None):
        self.dataname = dataname
        self.step = step
        self.case_mode = True
        if dataname is None:
            return 
        self._init()
    
    def _init(self, step):
        # self.data = Data(self.dataname)
        # self.model = WSLModel(self.dataname) # init
        # self.case_util = get_case_util(self.dataname, self.case_mode)
        # self.case_util.set_step(step)
        # # self.case_util.connect_model(self.model) 
        # self.model = self.case_util.create_model(VideoModel)
        self.model = VideoModel(self.dataname, step, self.case_mode)

    def reset(self, dataname, step=None):
        self.dataname = dataname
        self._init(step)

    def run_model(self):
        # self.model = self.case_util.run(use_buffer=False)
        self.model.run()

    # def update_model(self, constraints):
    #     if self.case_mode:
    #         # create a new model with step+1 and load pre-saved constraints
    #         step = self.model.step + 1
    #         # self._init(step) 
    #         self.model = VideoModel(self.dataname, step)
    #     else:
    #         self.model.update_constraints(constraints)
    #     self.run_model()

    def get_manifest(self):
        # res = self.case_util.get_base_config()
        # res["real_step"] = self.model.step
        res = {}
        res["class_name"] = self.model.data.classlist
        # res["label_consistency"] = res["parameters"][str(self.case_util.step)]["label_consistency"]
        # res["symmetrical_consistency"] = res["parameters"][str(self.case_util.step)]["symmetrical_consistency"]
        return res
    
    