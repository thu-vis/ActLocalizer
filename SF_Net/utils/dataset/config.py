'''
Author: Changjian Chen
Date: 2021-11-09 00:17:10
LastEditTime: 2022-01-04 21:59:17
LastEditors: Changjian Chen
Description: 
FilePath: /VideoVis/data/changjian/VideoModel/SF-Net/utils/dataset/config.py

'''

# config.py
import os.path
import sys
import numpy as np

# gets home dir cross platform
# HOME = os.path.expanduser("~")

HOME = os.path.dirname(sys.modules[__name__].__file__)
HOME = os.path.join(HOME, "../..")
HOME = os.path.normpath(HOME)

class Config(object):
    def __init__(self):
        # raw data
        self.raw_data_root = os.path.normpath(os.path.join(HOME, "../Data/"))
        
        # first-level directory
        self.data_root = os.path.normpath(os.path.join(HOME, "./data/"))
        self.result_root = os.path.normpath(os.path.join(HOME, "./result/"))
        
        # extension
        self.image_ext = ".jpg"
        self.mat_ext = ".mat"
        self.pkl_ext = ".pkl"

        # second-level directory
        self.beoid = "BEOID"
        self.gtea = "GTEA"
        self.thumos = "Thumos14"
        self.thumos19 = "Thumos19"
        self.thumosswin = "ThumosSwin"
        self.anomaly = 'Anomaly'

        basic_data_config = {
            "feature_dim": 1024,
            "tiou_thresholds": np.linspace(0.1, 0.7, 4),
            "train_subset": "training",
            "test_subset": "validation",
            "threshold_type": "mean",
            "stride": 16,
            "threshold_type": "mean",
            "prediction_filename": './data/prediction.json',
        }

        self.data_config = {
            self.beoid: {
                "fps": 30,
                "t_max": 100,
                "t_max_ctc": 400,
                "num_class": 34,
                "groundtruth_filename": os.path.join(self.data_root, "beoid_groundtruth.json"),
            },
            
            self.gtea:{
                "fps": 15,
                "t_max": 100,
                "t_max_ctc": 150,
                "num_class": 7,
                "groundtruth_filename": os.path.join(self.data_root, "gtea_groundtruth.json"),
            },

            self.thumos:{
                "fps": 25,
                "video_fps": 30,
                "t_max": 750,
                "t_max_ctc": 2800,
                "train_subset": "validation",
                "test_subset": "test",
                "num_class": 20,
                "groundtruth_filename": os.path.join(self.data_root, "th14_groundtruth.json")
            },

            self.thumos19:{
                "fps": 25,
                "video_fps": 30,
                "t_max": 750,
                "t_max_ctc": 2800,
                "train_subset": "validation",
                "test_subset": "test",
                "num_class": 19,
                "video_path": os.path.join(self.raw_data_root, "Thumos14/videos"),
                "frame_path": os.path.join(self.raw_data_root, "Thumos14/frames"),
                "groundtruth_filename": os.path.join(self.data_root, "th19_groundtruth.json")
            },
            self.thumosswin:{
                "fps": 30,
                "video_fps": 30,
                "t_max": 750,
                "t_max_ctc": 2800,
                "train_subset": "validation",
                "test_subset": "test",
                "num_class": 19,
                "video_path": os.path.join(self.raw_data_root, "Thumos14/videos"),
                "frame_path": os.path.join(self.raw_data_root, "Thumos14/frames"),
                "groundtruth_filename": os.path.join(self.data_root, "thswin_groundtruth.json")
            },
            self.anomaly:{
                "fps": 30,
                "video_fps": 30,
                "t_max": 100,
                "t_max_ctc": 400,
                "train_subset": "validation",
                "test_subset": "test",
                "num_class": 4,
                "groundtruth_filename": os.path.join(self.data_root, "anomaly_groundtruth.json")
            }
        }

        # assign basic data config
        for key in self.data_config:
            for basic_key in basic_data_config:
                if basic_key not in self.data_config[key]:
                    self.data_config[key][basic_key] = basic_data_config[basic_key]


        



config = Config()