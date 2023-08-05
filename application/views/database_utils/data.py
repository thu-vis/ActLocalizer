from asyncio.log import logger
import io
import os
from scipy import io as sio
from scipy import sparse
import sqlite3

from ..utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from ..utils.config_utils import config
from ..utils.log_utils import logger
from .video import VideoHelper
from time import time
import matplotlib.pyplot as plt
from IPython import embed
import numpy as np


def encoding_pair(i, j):
    return str(i) + "-" + str(j)

def decoding_pair(pair):
    a, b = pair.split("-")
    return int(a), int(b)

def feature_norm(features):
    norms = (features**2).sum(axis=1)
    norms = norms ** 0.5
    features = features / norms[:, np.newaxis]
    return features


class DatabaseLoader():
    '''
    base operation of database
    '''

    def __init__(self, dataname, step=0):
        # open database
        self.dataname = dataname
        self.step = step 
        self.common_data_root = os.path.join(config.data_root, dataname)
        self.step_data_root = os.path.join(self.common_data_root, "step" + str(step), )
        self.previous_data_root = None if step == 0 else os.path.join(self.common_data_root, "step" + str(step - 1), )
        # self.save_path = 'data/' + dataname + '/step_' + str(step)
        # self.meta_data = pickle_load_data(self.save_path + '/meta_info.pkl')
        # self.conn = sqlite3.connect(self.save_path + '/database.db', detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        # sqlite3.register_adapter(np.ndarray, self._adapt_array)
        # sqlite3.register_converter("array", self._convert_array)

    def get_feature_of_single_frame(self, video_id, frame_id):
        cursor = self.conn.cursor()
        cursor.execute('''  select feature
                            from data 
                            where video_id = {}
                            and frame_id = {}  '''.format(video_id, frame_id))
        fea_index = cursor.fetchone()[0]
        cursor.execute('''  select feature
                            from features
                            where id = {}   '''.format(fea_index))
        return cursor.fetchone()[0]
    
    def get_classification_of_single_frame(self, video_id, frame_id):
        cursor = self.conn.cursor()
        cursor.execute('''  select classification
                            from data
                            where video_id = {}
                            and frame_id = {}  '''.format(video_id, frame_id))
        return cursor.fetchone()[0]

    def get_action_score_of_single_frame(self, video_id, frame_id):
        cursor = self.conn.cursor()
        cursor.execute('''  select action_score
                            from data
                            where video_id = {}
                            and frame_id = {}  '''.format(video_id, frame_id))
        return cursor.fetchone()[0]
    
    def get_ground_truth_of_single_frame(self, video_id, frame_id):
        cursor = self.conn.cursor()
        cursor.execute('''  select groundtruth
                            from data
                            where video_id = {}
                            and frame_id = {}  '''.format(video_id, frame_id))
        return cursor.fetchone()[0]

    def _adapt_array(self, arr: np.ndarray):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return bytes(sqlite3.Binary(out.read()))

    def _convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

class Data(DatabaseLoader):
    def __init__(self, dataname, step=0, mode="pred"):
        # load some meta information
        super(Data, self).__init__(dataname, step)
        self.debug = False
        t0 = time()
        # fixed data when updating
        self.features = np.load(os.path.join(self.common_data_root, "all_features.npy"))
        self.meta = pickle_load_data(os.path.join(self.common_data_root, "meta_data.pkl"))

        # changed data when updating
        self.affinity_matrix = None # load when need
        # TODO: prediction is the output of Class Propagation
        # self.original_prediction = np.load(os.path.join(self.common_data_root, "classification_prediction.npy"))
        # self.prediction = np.load(os.path.join(self.step_data_root, "propagation.npy"))
        print("load processed data", time() - t0)
        self.gt_labels = self.meta["gt_labels"]
        self.single_frames = self.meta["single_frames"]
        self.single_frame_labels = self.meta['single_frame_labels']
        self.whole_to_indvdual = self.meta["whole_to_indvdual"]
        self.indvdual_to_whole = {self.whole_to_indvdual[i]: i for i in self.whole_to_indvdual}
        self.classlist = [i.decode() for i in self.meta["classlist"]]
        self.args = self.meta["args"]
        self.train_vid = self.meta["trainidx"]
        self.test_vid = self.meta["testidx"]
        self.all_vid = self.train_vid + self.test_vid
        self.videonames = self.meta["videonames"]
        self.videonames = [v.decode() + ".mp4" for v in self.videonames]

        self.raw_video_root = os.path.join(self.common_data_root, "videos")
        if not os.path.exists(self.raw_video_root):
            self.raw_video_root = "/data/changjian/VideoModel/Data/Thumos14/videos"
        self.video_buffer_root = os.path.join(self.raw_video_root, "buffer")
        check_dir(self.video_buffer_root)
        self.video_pose_root = os.path.join(self.raw_video_root, "pose", "openpose")

        self.video = None
        # for debug
        # self.video = VideoHelper(0, os.path.join(config.raw_video_root, \
        #     self.videonames[self.train_vid[0]]), self.args.fps, self.args.stride)

        # process single frames
        self.single_frames_id_map = {}
        self.single_frames_vid_fid_map = {}
        for single_idx, single_frame in enumerate(self.single_frames):
            single_frame["single_idx"] = single_idx
            self.single_frames_id_map[single_idx] = single_frame
            self.single_frames_vid_fid_map[single_frame["id"]] = single_frame

    def run(self):
        None
    
    def get_single_frame_by_single_idx(self, idx):
        return self.single_frames_id_map[idx]

    def get_single_frame_by_vid_fid(self, vfid):
        return self.single_frames_vid_fid_map[vfid]

    def get_all_features(self):
        features = self.features.copy()
        features = feature_norm(features)
        return features
    
    def get_ground_truth_labels(self):
        return self.gt_labels.copy()
    

    def get_video_and_frame_id_by_idx(self, idx):
        res = self.whole_to_indvdual[idx]
        return decoding_pair(res)
    
    def get_idx_by_video_and_frame_id(self, v_id, f_id):
        code = encoding_pair(v_id, f_id)
        return self.indvdual_to_whole[code]


    def get_single_frame(self, vid, fid):
        # assert vid == self.video.vid
        if self.video is None or self.video.vid != vid:
            vname = os.path.join(self.raw_video_root, \
                self.videonames[vid])
            vbuffer = os.path.join(self.video_buffer_root, \
                self.videonames[vid])
            vbuffer = os.path.splitext(vbuffer)[0]
            vpose = os.path.join(self.video_pose_root, \
                self.videonames[vid] + "-pose.pkl")
            self.video = VideoHelper(vid, vname, vbuffer, vpose, self.args.fps, self.args.stride)
        else:
            # logger.info("the same video, skip VideoHelper init")
            None
        return self.video.get_single_frame(fid)
    
    def get_video(self, vid):
        return os.path.join(self.raw_video_root, self.videonames[vid])
