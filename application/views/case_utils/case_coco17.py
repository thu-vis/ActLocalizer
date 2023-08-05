import numpy as np
import os
import json

from .case_base import CaseBase
from ..utils.config_utils import config
from ..utils.helper_utils import pickle_save_data, pickle_load_data
from ..utils.log_utils import logger

# # What to do in each step
# step0: before and after word embedding finetuning
# step1: add rules and image embeddings through vis
# step2: improve constraint weights // pattern: 有很多区域被检测出来了，但是conf不高
# step3: look the distribution and validation //pattern：有些区域既有label错了，又有image错了
# step4: 

class CaseCOCO17(CaseBase):
    def __init__(self, case_mode):
        dataname = config.coco17
        super(CaseCOCO17, self).__init__(dataname, case_mode)
        self.step = self.base_config["step"]

    def run(self, use_buffer=False):
        # if step and self.case_mode:
        #     raise ValueError("case_mode is set but step is provided")
        # if self.step == 0:
        #     None
        # elif self.step >= 1:
        if use_buffer and self.model.buffer_exist():
            logger.info("buffer exists. Loading model.")
            # self.model.load_model()
            dataname = self.model.dataname
            self.model = pickle_load_data(self.model.buffer_path)
            self.model.update_data_root(dataname, self.step)
            self.model._init_data()
            return self.model
        # logger.info("You do not intend to use the model buffer or the buffer does not exist")
        if use_buffer:
            logger.info("The model buffer does not exist. Run the model.")
        else:
            logger.info("You do not intend to use the model buffer. Rerun the model.")
        self.model.run()
        self.model.save_model()
        return self.model