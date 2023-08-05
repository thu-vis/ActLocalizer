import numpy as np
import os
import json
from time import time

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE

from ..utils.log_utils import logger
from ..utils.config_utils import config
from ..utils.helper_utils import json_load_data, json_save_data
from ..utils.helper_utils import pickle_save_data, pickle_load_data, check_dir
from ..database_utils.data import Data
# from ..database_utils.data import Data

from .cluster_helper import ClusterHelper
from .frame_selector import FrameSelector
from .propagation import MatchAndProp


class VideoModel(object):
    def __init__(self, dataname=None, step=0, case_mode=True):
        self.dataname = dataname
        self.step = step
        self.case_mode = case_mode
        # if config:
        #     self.config = config
        # else:
        #     self.config = {
        #         "step": 1,
        #         "text_k": 9,
        #         "image_k": 9,
        #         "pre_k": 100,
        #         "weight": 1
        #     }
        self.nearest_actions = 3
        self.nearest_frames = 4
        self.window_size = 5
        if dataname is None:
            return 
        self._init()
    
    def update_data_root(self, dataname, step):
        self.step = step
        suffix = "step" + str(step)
        self.common_data_root = os.path.join(config.data_root, dataname)
        self.step_data_root = os.path.join(config.data_root, self.dataname, suffix)
        self.previous_data_root = None if step == 0 else os.path.join(config.data_root, self.dataname, "step" + str(step - 1))
        self.buffer_path = os.path.join(self.step_data_root, "model.pkl")

    def _init(self):
        logger.info("current config of model: dataname-{}, step-{}".format(self.dataname, self.step))
        # data 
        self.update_data_root(self.dataname, self.step)

        self.data = Data(self.dataname, self.step)
        logger.info("finished load data")
        self.prop_model = MatchAndProp(self.data)
        # if self.case_mode:
        #     self.prop_model.load_pre_saved_constraints()
        logger.info("finished load match and prop")

        self.cluster_helper = ClusterHelper(self.prop_model)
        logger.info("finished cluster")
        self.frame_selector = FrameSelector(self.prop_model)
        logger.info("finished frame selector")
        

    def reset(self, dataname, step):
        self.dataname = dataname
        self.step = step
        self._init()

    def run(self):
        if self.case_mode:
            self.data.run() # TODO: change the name to load data
            self.prop_model.load_prerun_result()
            self.cluster_helper.load_prerun_result()

        else:
            self.data.run()
            self.prop_model.run()
            self.cluster_helper.run()

    def update_constraints(self, constraints):
        if self.case_mode:
            if self.step < 9:
                cur_index = self.cluster_helper.cur_index
                self.reset(self.dataname, self.step + 1)
                self.run()
                if cur_index != -1:
                    self.cluster_helper.load_hierarchy_i(cur_index)
                return True
            return False

        self.prop_model.add_constraint(constraints)
        self.prop_model.run()
        prop_data = self.prop_model.get_data_for_next()
        save_data = self.cluster_helper.update()

        self.reset(self.dataname, self.step + 1)
        self.prop_model.save_data_for_next(prop_data)
        self.cluster_helper.load_from_predecessor(save_data)
        return True

    def remove_action(self, action_id):
        self.cluster_helper.remove(action_id)
        return self.update_constraints({})

    def recommend_by_prediction(self, cdx, action_id, labeled_frame_id, pos):
        """
        recommend by prediction 
        return: 
            result - true/false
            recommfid - int
            prediction - []
        """
        action_info = action_id.split('-')
        video_id, single_fid = int(action_info[0]), int(action_info[1])
        labeled_fid = int(labeled_frame_id.split('-')[1])
        if pos < 0:
            label_mode = "left"
        else:
            label_mode = "right"

        if self.case_mode:
            if self.step == 0 and action_id == "360-279" or \
                self.step == 1 and action_id == "360-164":
                recom_path = os.path.join(self.step_data_root, "recommendation", action_id + "_recommendation.json")
                recom = json_load_data(recom_path)
                predictions = recom["pred_scores"]
                start_fid = recom["left_bound"]

                if label_mode == "left":
                    recom_fid = recom["left_bound"]
                    self.prop_model.set_pred_scores_of_video_with_given_boundary(video_id, cdx, recom_fid, 
                        single_fid, predictions[recom_fid - start_fid: single_fid - start_fid + 1])

                else:
                    recom_fid = recom["right_bound"]
                    self.prop_model.set_pred_scores_of_video_with_given_boundary(video_id, cdx, single_fid, 
                        recom_fid, predictions[single_fid - start_fid: recom_fid - start_fid + 1])

                if "information" in recom:
                    info = recom["information"]
                    self.prop_model.set_pred_scores_of_video_with_given_boundary(video_id, cdx, info["begin"], 
                        info["begin"] + len(info["score"]) - 1, info["score"])

                ret = {
                    "recom_pos": recom_fid,
                    "recom_direct": label_mode,
                    "label_id": labeled_fid
                }
                return True, ret
        else:
            return False, {}


    def get_hierarchy_meta_data(self):
        return self.cluster_helper.load_hierarchy_meta()

    def get_hierarchy(self, h_id):
        return self.cluster_helper.load_hierarchy_i(h_id)    

    def get_rep_frames(self, vid, bound, target):
        return self.frame_selector.select_frames(vid, bound[0], bound[1], target)
    
    def get_alignment_of_anchor_action(self, cls, aid):
        return self.cluster_helper.get_alignment_of_anchor_action(cls, aid, self.nearest_actions)
    
    def get_pred_scores_of_video_with_given_boundary(self, vid, cls, start, end):
        return self.prop_model.get_pred_scores_of_video_with_given_boundary(vid, cls, start, end)

    def save_model(self, path=None):
        logger.info("save model buffer")
        buffer_path = self.buffer_path
        if path:
            buffer_path = path
        tmp_data = self.data
        tmp_video = self.data.video
        self.data.video = None
        self.data = None
        pickle_save_data(buffer_path, self)
        self.data = tmp_data

    def load_model(self, path=None):
        buffer_path = self.buffer_path
        if path:
            buffer_path = path
        self = pickle_load_data(buffer_path)
        self.data = Data(self.dataname, self.step)
    
    def buffer_exist(self, path=None):
        buffer_path = self.buffer_path
        if path:
            buffer_path = path
        logger.info(buffer_path)
        if os.path.exists(buffer_path):
            return True
        else:
            return False
    
    