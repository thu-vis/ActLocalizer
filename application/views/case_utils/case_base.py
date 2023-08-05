import numpy as np
import os
from ..utils.helper_utils import json_load_data, pickle_load_data
from ..utils.helper_utils import pickle_save_data, json_save_data
from ..utils.config_utils import config
from ..utils.log_utils import logger

class CaseBase():
    def __init__(self, dataname, case_mode):
        self.dataname = dataname
        self.case_mode = case_mode
        
        self.model = None
        self.base_config = None
        self.step = 0

        self.pred_result = {}

        self._load_base_config()
        
    def connect_model(self, model):
        self.model = model

    def create_model(self, model):
        self.model = model(self.dataname, self.step, self.base_config)
        return self.model
    
    def set_step(self, step):
        if step:
            self.step = step
            logger.info("*************** current step: {} ************".format(step))

    def _load_base_config(self):
        json_data = json_load_data(os.path.join(config.case_util_root, "case_config.json"))
        try:
            self.base_config = json_data[self.dataname]
        except:
            self.base_config = {
                "step": 0,
            }

    def get_base_config(self):
        return self.base_config

    def load_data(self, name):
        # TODO:
        return None