# Convert data into database
# Other Statistical AIDS

import io
import os, shutil
from math import floor, ceil

import numpy as np
import sys

import pandas as pd
from tqdm import tqdm
from time import time
import argparse
import sqlite3
from application.views.utils.helper_utils import pickle_save_data

parser = argparse.ArgumentParser(
    description='test')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--step', default='0',
                    type=str, help='')
parser.add_argument('--dataname', default='thumos19',
                    type=str, help='')
args = parser.parse_args()

class DataConverter():
    def __init__(self, dataset, step):
        self.dataset = dataset
        self.step = step
        self.save_path = '../../../data/' + args.dataname + '/step_' + args.step
        self.info_path = '../../../test/' + args.dataname
        self.meta_data = {}
        self.trans = {}
        self.back_index = 19
        self.anns_filename = "THUMOS2.txt"

    def convert(self):
        # Save meta data in meta_info.pkl
        self._generate_meta_data()
        pickle_save_data(self.save_path+'/meta_info.pkl', self.meta_data)
        # save data in database.db
        self._save_data_in_database()
        print("Convert successfully!")

    # generate dataset meta data
    def _generate_meta_data(self):
        # Generate meta data, which includes:
        #   - classes: all class names
        #   - class_2_id_map: class names to class id
        #   - video_names: names of videos, e.g., ['video_name1', 'video_name2']
        #   - num_frames: number of frames for each videos, e.g., [123, 345]
        #   - annotated_frames: annotated frames ids, e.g., [[2,5,8], [12,45,21]]
        #   - train_idx: ids of all training videos, e.g., [0, 1]
        #   - add_info: other information

        # generate save path
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(self.save_path)

        # check info path
        if not os.path.exists(self.info_path):
            raise Exception('Convert: information not exist.')

        ann_path = self.info_path + '/groundtruth/' + self.dataset + '-Annotations'
        clas_path = ann_path + '/classlist.npy'

        if args.dataname == 'thumos':
            ann_path = self.info_path + '/groundtruth/Thumos14-Annotations'
            clas_path = ann_path + '/classlist_20classes.npy'

        # generate meta data
        meta_data = {}
        meta_data['dataset'] = self.dataset
        meta_data['stride'] = 16

        classes, cdx, class_2_id_map = [], 0, {}
        np_classlist = np.load(clas_path)
        for cname in np_classlist:
            cname = cname.decode('utf-8')
            classes.append(cname)
            class_2_id_map[cname] = cdx
            cdx += 1
        classes.append("Background")
        class_2_id_map["Background"] = cdx
        cdx += 1
        del np_classlist
        meta_data['classes'] = classes
        meta_data['class_2_id_map'] = class_2_id_map

        video_names, num_frames, annotated_frames = [], [], []
        np_videoname = np.load(ann_path + '/videoname.npy')
        np_labels = np.load(ann_path + '/labels.npy', allow_pickle=True)
        np_subsets = np.load(ann_path + '/subset.npy', allow_pickle=True)
        np_durations = np.load(ann_path + '/duration.npy')
        fps, stride = 25, 16

        for i in range(np_labels.shape[0]):
            if len(np_labels[i]) > 0:
                if np_subsets[i].decode('utf-8') == 'validation':
                    vname = np_videoname[i].decode('utf-8')
                    video_names.append(vname)
                    out_path = self.info_path + "/outputs/" + vname + "/"
                    nframe = min(np.load(out_path + "score.npy").shape[0] * stride, round(np_durations[i][0] * fps))
                    num_frames.append(nframe)
                    self.trans[vname] = i
        del np_videoname, np_labels, np_subsets
        meta_data['video_names'] = video_names
        meta_data['num_frames'] = num_frames

        filename = self.anns_filename
        ann_data = pd.read_csv(ann_path + '/single_frames/' + filename, names=[
            'vid', 'time', 'label'], converters={'vid': self._strip, 'time': self._make_float, 'label': self._strip})
        for video_name in video_names:
            time_list = ann_data[ann_data.vid == video_name][['time']].to_numpy().tolist()
            frame_list = []
            for time in time_list:
                frame_list.append(round(time[0] * fps))
            annotated_frames.append(frame_list)
        meta_data['annotated_frames'] = annotated_frames
        del ann_data

        meta_data['train_idx'] = list(range(len(video_names)))
        meta_data['add_info'] = {}
        self.meta_data = meta_data

    # save data in database.db
    def _save_data_in_database(self):
        # and save data in database.db, which includes a table with:
        #   - video_id
        #   - frame_id
        #   - classification
        #   - action_score
        #   - groundtruth
        #   - feature

        # get feature path
        fea_path = self.info_path + '/features/' + self.dataset + '-Features.npy'
        features = np.load(fea_path, allow_pickle=True)

        ann_path = self.info_path + '/groundtruth/' + self.dataset + '-Annotations'
        if args.dataname == 'thumos':
            ann_path = self.info_path + '/groundtruth/Thumos14-Annotations'
        seg_list = np.load(ann_path + '/segments.npy', allow_pickle=True)
        label_list = np.load(ann_path + '/labels.npy', allow_pickle=True)
        fps, stride = 25, 16

        # set data adapt

        def adapt_array(arr: np.ndarray):
            out = io.BytesIO()
            np.save(out, arr)
            out.seek(0)
            return bytes(sqlite3.Binary(out.read()))

        def convert_array(text):
            out = io.BytesIO(text)
            out.seek(0)
            return np.load(out)

        # connect database
        conn = sqlite3.connect(self.save_path + '/database.db', detect_types=sqlite3.PARSE_DECLTYPES)
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("array", convert_array)
        c = conn.cursor()
        # create table data、features
        c.execute('''create table data
                    (video_id       int     not null,
                     frame_id       int     not null,
                     classification array,
                     action_score   real,
                     groundtruth    int,
                     feature        int,
                     primary key (video_id, frame_id)
                     )''')
        c.execute('''create table features
                    (id         int       primary key     not null,
                     feature    array
                     )''')
        # solve video and insert data
        data_insert = "insert into data values(?,?,?,?,?,?)"
        fea_insert = "insert into features (id, feature) values(?,?)"
        fea_index = 0
        fea_count = 0
        for vid in self.meta_data["train_idx"]:
            vname = self.meta_data["video_names"][vid]
            vframes = self.meta_data["num_frames"][vid]
            idx = self.trans[vname]
            grth = [self.back_index] * vframes
            segs = seg_list[idx]
            labels = label_list[idx]
            vf_index = 0
            for i in range(len(segs)):
                seg = segs[i]
                label = labels[i]
                start = round(seg[0] * fps)
                end = round(seg[1] * fps)
                index = start
                while index <= end and index < vframes:
                    tmp = self.meta_data["class_2_id_map"][label]
                    if tmp < grth[index]:
                        grth[index] = tmp
                    index += 1
            out_path = self.info_path + "/outputs/" + vname + "/"
            tcam = np.load(out_path + "tcam.npy")
            score = np.load(out_path + "score.npy")
            for fid in range(vframes):
                cls = tcam[floor(fid/stride)]
                acs = float(score[floor(fid/stride)])
                gt = grth[fid]
                fea = fea_index
                c.execute(data_insert, (vid, fid, cls, acs, gt, fea, ))
                fea_count += 1
                if fea_count >= stride:
                    c.execute(fea_insert, (fea_index, features[vid][vf_index], ))
                    fea_index += 1
                    vf_index += 1
                    fea_count = 0
            if fea_count != 0:
                c.execute(fea_insert, (fea_index, features[vid][vf_index],))
                fea_index += 1
                fea_count = 0

        conn.commit()
        conn.close()

    # get background ratio
    def cal_background_ratio(self):
        ann_path = self.info_path + '/groundtruth/' + self.dataset + '-Annotations'
        if args.dataname == 'thumos':
            ann_path = self.info_path + '/groundtruth/Thumos14-Annotations'

        np_duration = np.load(ann_path + '/duration.npy')
        np_segments = np.load(ann_path + '/segments.npy', allow_pickle=True)
        ratios, ratio = [], 0
        time, nbktime = 0, 0
        for i in range(np_duration.shape[0]):
            time_i = np_duration[i][0]
            segs = np_segments[i]
            nbktime_i = 0
            if len(segs) == 0:
                continue
            for seg in segs:
                nbktime_i += (seg[1] - seg[0])
            ratios.append(1 - nbktime_i / time_i)
            time += time_i
            nbktime += nbktime_i
        ratio = 1 - nbktime / time
        return np.mean(ratios), ratio

    # calculate missing annotations
    def cal_missing_anns(self):
        ann_path = self.info_path + '/groundtruth/' + self.dataset + '-Annotations'
        if args.dataname == 'thumos':
            ann_path = self.info_path + '/groundtruth/Thumos14-Annotations'

        np_videonames = np.load(ann_path + '/videoname.npy')
        np_subsets = np.load(ann_path + '/subset.npy')
        np_segments = np.load(ann_path + '/segments.npy', allow_pickle=True)

        filename = 'THUMOS4.txt'
        ann_data = pd.read_csv(ann_path + '/single_frames/' + filename, names=[
            'vid', 'time', 'label'], converters={'vid': self._strip, 'time': self._make_float, 'label': self._strip})

        miss_list = []
        for i in range(np_videonames.shape[0]):
            if len(np_segments[i])==0:
                continue
            if np_subsets[i].decode('utf-8') != "validation":
                continue
            vname = np_videonames[i].decode('utf-8')
            time_list = ann_data[ann_data.vid == vname][['time']].to_numpy().tolist()
            segs = np_segments[i]
            count = 0
            for seg in segs:
                anno = False
                for time in time_list:
                    if time[0] >= seg[0] and time[0] <= seg[1]:
                        anno = True
                        break
                if not anno:
                    count += 1
            miss_list.append(count)
        return miss_list

    # utils for read pd
    def _strip(self, text):
        try:
            return text.strip()
        except AttributeError:
            return text

    def _make_float(self, text):
        return float(text.strip())

if __name__ == "__main__":
    data_converter = DataConverter(args.dataname, args.step)
    data_converter.convert()
    '''
    miss_list = data_converter.cal_missing_anns()
    sum = 0
    for miss in miss_list:
        sum += miss
    print(len(miss_list), sum, miss_list)
    '''