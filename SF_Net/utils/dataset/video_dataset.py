import os
import numpy as np
import glob
# import utils
import time
import json
import copy
from .config import config
from ..utils import check_dir, pickle_load_data, pickle_save_data, strlist2multihot, strlist2indlist, process_feat

try:    
    import torch
    from torch.utils import data
except:
    None

np_load_old = np.load
np_save_old = np.save

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
np.save = lambda *a, **k: np_save_old(*a, allow_pickle=True, **k)

def left_active(frame_label, frame_idx, label):
    if label not in frame_label[frame_idx]:
        frame_label[frame_idx].append(label)
    start = frame_idx - 1
    if start < 0:
        return 0
    for idx in range(start, -1, -1):
        if label not in frame_label[idx]:
            return idx + 1
    return 0

def right_active(frame_label, frame_idx, label):
    if label not in frame_label[frame_idx]:
        frame_label[frame_idx].append(label)
    max_idx = len(frame_label) - 1
    start = frame_idx + 1
    if start > max_idx:
        return max_idx
    for idx in range(start, max_idx + 1):
        if label not in frame_label[idx]:
            return idx - 1
    return max_idx

def distance_bewteen_an_interval_and_an_point(interval, point):
    if point >= interval[0] and point <= interval[1]:
        m = min(abs(interval[0] - point), abs(interval[1] - point))
        return -m
    else:
        m = min(abs(interval[0] - point), abs(interval[1] - point))
        return m

class Dataset():
    def __init__(self, args, groundtruth_file=None, train_subset='validation', \
        test_subset='test', preprocess_feat=False, mode='weakly', use_sf=True, \
        choice=1):
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.dataset_name = args.dataset_name
        self.mode = mode
        #  self.num_class = args.num_class
        self.choice = choice
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(
            args.feature_path, self.dataset_name + '-I3D-JOINTFeatures.npy')
        if self.dataset_name in [config.thumosswin]:
            self.path_to_features = os.path.join(
                args.feature_path, self.dataset_name + '-Features.npy')
        self.path_to_annotations = os.path.join(
            args.feature_path, self.dataset_name + '-Annotations')
        self.features = np.load(self.path_to_features, encoding='bytes')
        self.segments = np.load(os.path.join(
            self.path_to_annotations, 'segments.npy'))
        self.gtlabels = np.load(os.path.join(
            self.path_to_annotations, 'labels.npy'))
        self.labels = np.load(os.path.join(self.path_to_annotations,
                                           'labels_all.npy'))  # Specific to Thumos14
        self.fps = args.fps
        if groundtruth_file:
            with open(groundtruth_file, 'r') as fr:
                self.gt_info = json.load(fr)['database']
        else:
            self.gt_info = {}
        self.stride = args.stride
        if self.dataset_name == 'Thumos14':
            self.classlist20 = np.load(os.path.join(self.path_to_annotations,
                                                    'classlist_20classes.npy'))
        self.classlist = np.load(os.path.join(
            self.path_to_annotations, 'classlist.npy'))
        self.subset = np.load(os.path.join(
            self.path_to_annotations, 'subset.npy'))
        self.duration = np.load(os.path.join(
            self.path_to_annotations, 'duration.npy'))
        self.videoname = np.load(os.path.join(
            self.path_to_annotations, 'videoname.npy'))
        self.seed = args.seed
        self.increase = args.increase
        self.lst_valid = None
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.currentvalidx = 0
        self.labels_multihot = [
            strlist2multihot(labs, self.classlist) for labs in self.labels
        ]
        self.train_test_idx()
        self.classwise_feature_mapping()
        self.labels101to20 = None
        if self.dataset_name == 'Thumos14':
            self.labels101to20 == np.array(self.classes101to20())
        self.class_order = self.get_class_id()
        self.count_labels = self.get_count()
        np.random.seed(self.seed)
        self.all_frame_labels = self.get_all_frame_labels()
        self.active_labels = None
        if mode == 'weakly' or mode == 'fully':
            self.init_frame_labels = self.get_all_frame_labels()
        elif mode == 'single':
            if use_sf:
                self.init_frame_labels = self.get_labeled_frame_labels(
                    os.path.join(self.path_to_annotations, 'single_frames'))
            else:
                self.init_frame_labels = self.get_rand_frame_labels()
        elif mode == "active":
            if use_sf:
                # self.init_frame_labels = self.get_labeled_frame_labels(
                #     os.path.join(self.path_to_annotations, 'single_frames'))
                self.active_ratio = args.active_ratio
                self.init_frame_labels = self.get_active_labels(args.active_ratio)
            else:
                self.init_frame_labels = self.get_rand_frame_labels()
            # self.active_labels = self.random_sampling_actions(args.active_ratio)
        else:
            raise ValueError('wrong mode setting')

        #  self.init_frame_labels = self.get_mid_frame_labels()
        #  self.init_frame_labels = self.get_midc_frame_labels()
        #  self.init_frame_labels = self.get_bgmid_frame_labels()
        #  self.init_frame_labels  =self.get_frame_labels_custom_distribution()
        #  self.init_frame_labels = self.get_start_frame_labels(0.5)
        #  self.init_frame_labels = self.get_bgrand_frame_labels()
        #  self.init_frame_labels = self.get_bgall_frame_labels()
        self.frame_labels = copy.deepcopy(self.init_frame_labels)
        # self.single_frame_labels = self.get_labeled_frame_labels(
        #             os.path.join(self.path_to_annotations, 'single_frames'))
        # if mode == "active":
        #     self.frame_labels = self.combine_active_labels(self.frame_labels, \
        #         self.active_labels)
        self.clusters = self.init_clusters()
        self.num_frames = np.sum([
            np.sum([len(p) for p in self.frame_labels[i] if len(p) > 0])
            for i in self.trainidx
        ])
        
    def combine_active_labels(self, frame_labels, active_labels):
        for idx in self.trainidx:
            active_label = active_labels[idx]
            for ac in active_label:
                s = ac[0]
                e = min(ac[1], len(frame_labels[idx]))
                if self.increase:
                    e = min(ac[1] + 1, len(frame_labels[idx]))
                l = ac[2]
                for f in range(s, e):
                    if l not in frame_labels[idx][f]:
                        frame_labels[idx][f].append(l)
        return frame_labels
    

    def update_frame_label(self, idx, label):
        if self.mode == "active":
            # active_label = self.active_labels[idx]
            # for ac in active_label:
            #     s = ac[0]
            #     # e = min(ac[1] + 1, len(self.frame_labels[idx]))
            #     e = min(ac[1], len(self.frame_labels[idx]))
            #     l = ac[2]
            #     for f in range(s, e):
            #         if l not in self.frame_labels[idx][f]:
            #             self.frame_labels[idx][f].append(l)
            for i in range(len(self.frame_labels[idx])):
                res = label[i] + self.active_labels[idx][i]
                res = list(set(res))
                res.sort()
                self.frame_labels[idx][i] = res
        else:        
            self.frame_labels[idx] = label
        return True


    def is_actively_labeled(self, idx, frame_id):
        if self.active_labels is None:
            return False
        active_label = self.active_labels[idx]
        label = active_label[frame_id]
        if len(label) > 0:
            return True
        return False

    def read_single_frame_data(self, annotation_dire):
        import pandas as pd

        def strip(text):
            try:
                return text.strip()
            except AttributeError:
                return text

        def make_float(text):
            return float(text.strip())
        # for filename in os.listdir(annotation_dire):
        #     data = pd.read_csv(os.path.join(annotation_dire, filename), names=[
        #                        'vid', 'time', 'label'], converters={'vid': strip, 'time': make_float, 'label': strip})
        #     datas.append(data)
        single_anno_file = self.dataset_name.strip("123456789").upper() + \
            str(self.choice) + ".txt"
        if self.dataset_name in [config.thumosswin]:
            single_anno_file = "THUMOS".upper() + \
            str(self.choice) + ".txt"
        print("choiced file", single_anno_file)
        data = pd.read_csv(os.path.join(annotation_dire, single_anno_file), names=[ \
            'vid', 'time', 'label'], converters={'vid': strip, 'time': make_float, 'label': strip})
        return data


    def get_labeled_frame_labels(self, annotation_dire):
        data = self.read_single_frame_data(annotation_dire)
        labels = []
        classlist = self.get_classlist()
        for i in range(len(self.videoname)):
            # 修改为只用某一个标注
            # data = datas[self.choice]
            # data = datas[np.random.choice(range(len(datas)))]
            max_len = len(self.features[i])
            frame_label = [[] for _ in range(max_len)]
            if i not in self.trainidx:
                labels += [frame_label]
                continue
            vname = self.videoname[i].decode('utf-8')
            fps = self.get_fps(i)
            time_class = data[data.vid == vname][['time', 'label']].to_numpy()
            for t, c in time_class:
                pos = int(t * fps / self.stride)
                if pos >= max_len:
                    continue
                intl = strlist2indlist([c], classlist)[0]
                if intl not in frame_label[pos]:
                    frame_label[pos].append(intl)
            labels += [frame_label]
        return labels

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == self.train_subset and len(self.gtlabels[i]) > 0:
                #  if s.decode('utf-8') == train_str:
                self.trainidx.append(i)
            elif s.decode('utf-8') == self.test_subset and len(self.gtlabels[i]) > 0:
                self.testidx.append(i)

    def classwise_feature_mapping(self):

        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode('utf-8'):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self, is_training=True):

        if is_training == True:
            features = []
            labels = []
            idx = []

            # random sampling
            rand_sampleid = np.random.choice(len(self.trainidx),
                                             size=self.batch_size)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])

            count_labels = np.array([self.count_labels[i] for i in idx])
            if self.labels101to20 is not None:
                count_labels = count_labels[:, self.labels101to20]
            features = np.array(
                [process_feat(self.features[i], self.t_max) for i in idx])
            video_labels = np.array([self.labels_multihot[i] for i in idx])

            return features, video_labels, count_labels

            return np.array([
                process_feat(self.features[i], self.t_max) for i in idx
            ]), np.array([self.labels_multihot[i] for i in idx]), count_labels

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]

            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1

            return np.array([feat]), np.array(labs), done

    def get_feature(self, idx):
        return copy.deepcopy(self.features[idx])

    def get_vname(self, idx):
        return self.videoname[idx].decode('utf-8')

    def get_duration(self, idx):
        return self.duration[idx]

    def get_init_frame_label(self, idx):
        return copy.deepcopy(self.init_frame_labels[idx])
    
    def get_single_label(self, idx):
        if self.mode == "active":
            return copy.deepcopy(self.single_frame_labels[idx])
        return copy.deepcopy(self.init_frame_labels[idx])

    def get_frame_data(self):
        features = []
        labels = []
        one_hots = np.eye(len(self.get_classlist()))
        for idx in self.trainidx:
            feature = self.get_feature(idx)
            frame_label = self.get_frame_label(idx)
            assert len(feature) == len(frame_label)
            for i, ps in enumerate(frame_label):
                if len(ps) < 1:
                    continue
                else:
                    for p in ps:
                        features += [feature[i]]
                        labels += [one_hots[p]]
        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    def get_frame_label(self, idx):
        return copy.deepcopy(self.frame_labels[idx])

    def get_fps(self, idx):
        vname = self.videoname[idx].decode('utf-8')
        try:
            fps = self.gt_info[vname].get('fps', self.fps)
        except:
            fps = self.fps
        return fps

    def get_video_label(self, idx, background=False):
        video_label = np.concatenate(self.all_frame_labels[idx]).astype(int)
        video_label = list(set(video_label))
        return video_label

    def get_gt_frame_label(self, idx):
        return copy.deepcopy(self.all_frame_labels[idx])

    def get_trainidx(self):
        return copy.deepcopy(self.trainidx)

    def get_testidx(self):
        return copy.deepcopy(self.testidx)

    def get_segment(self, idx):
        return self.segments[idx]

    def get_classlist(self):
        if self.dataset_name == 'Thumos14':
            return self.classlist20
        else:
            return self.classlist

    def get_frame_counts(self):
        #  counts = np.sum([len(np.where(self.frame_labels[i] != -1)[0]) for i in self.trainidx])
        return self.num_frames

    def update_num_frames(self):
        self.num_frames = np.sum([
            np.sum([1 for p in self.frame_labels[i] if len(p) > 0])
            for i in self.trainidx
        ])

    def classes101to20(self):

        classlist20 = np.array([c.decode('utf-8') for c in self.classlist20])
        classlist101 = np.array([c.decode('utf-8') for c in self.classlist])
        labelsidx = []
        for categoryname in classlist20:
            labelsidx.append([
                i for i in range(len(classlist101))
                if categoryname == classlist101[i]
            ][0])

        return labelsidx

    def get_class_id(self):
        # Dict of class names and their indices
        d = dict()
        for i in range(len(self.classlist)):
            k = self.classlist[i]
            d[k.decode('utf-8')] = i
        return d

    def get_count(self):
        # Count number of instances of each category present in the video
        count = []
        num_class = len(self.class_order)
        for i in range(len(self.gtlabels)):
            gtl = self.gtlabels[i]
            cnt = np.zeros(num_class)
            for j in gtl:
                cnt[self.class_order[j]] += 1
            count.append(cnt)
        count = np.array(count)
        return count

    def init_clusters(self):
        clusters = [[] for _ in range(len(self.frame_labels))]
        for idx in self.trainidx:
            frame_label = self.get_init_frame_label(idx)
            for jdx, pls in enumerate(frame_label):
                if len(pls) < 1:
                    continue
                for pl in pls:
                    clusters[idx].append([jdx, jdx, jdx, pl])
        return clusters

    def get_all_actions(self):
        # if os.path.exists(filepath):
        #     return np.load(filepath).tolist()
        all_actions = []
        for idx in self.trainidx:
            seg = self.segments[idx]
            gtl = self.gtlabels[idx]
            sgl = self.single_labels[idx]
            for i in range(len(seg)):
                all_actions.append([idx] + seg[i] + [gtl[i]] + [sgl[i]] + [False])
        self.all_actions = all_actions

    def random_sampling_actions(self, active_ratio):
        label_dir = os.path.join(self.path_to_annotations, "active_label")
        check_dir(label_dir)
        filepath = os.path.join(label_dir, str(active_ratio) + "_ratio.npy")
        all_actions = self.all_actions
        labeled_actions_num = int(len(all_actions) * active_ratio)
        idxs = np.array(range(len(all_actions)))
        np.random.shuffle(idxs)
        for idx in idxs[:labeled_actions_num]:
            self.all_actions[idx][5] = True
        active_labels = []
        classlist = self.get_classlist()
        for i in range(len(self.videoname)):
            max_len = len(self.features[i])
            frame_label = [[] for _ in range(max_len)]
            active_labels += [frame_label]
        single_frame_labels = copy.deepcopy(active_labels)
        
        for i in range(len(self.all_actions)):
            action = self.all_actions[i]
            fps = self.get_fps(action[0])
            vid = action[0]
            frame_label = active_labels[vid]
            max_len = len(frame_label)
            intl = strlist2indlist([action[3]], classlist)[0]
            if action[5]: # labeled
                frame_label = active_labels[vid]
                start = action[1] * fps / self.stride
                end = action[2] * fps / self.stride
                if end < start:
                    continue
                elif end - start < 1.0:
                    end += 1
                start = max(0, int(start))
                if self.increase:
                    end = min(max_len, int(end + 1))
                else:
                    end = min(max_len, int(end))
                for pid in range(start, end):
                        if intl not in frame_label[pid]:
                            frame_label[pid].append(intl)
            else: # single frame
                frame_label = single_frame_labels[vid]
                for pos in action[4]:
                    pos = int(pos * fps / self.stride)
                    if pos >= max_len:
                        continue
                    if intl not in frame_label[pos]:
                        frame_label[pos].append(intl)
        return active_labels, single_frame_labels
        # res = [all_actions[i] for i in idxs[:labeled_actions_num]]
        # active_labels = [[] for _ in range(len(self.segments))]
        # classlist = self.get_classlist()
        # for r in res:
        #     fps = self.get_fps(r[0])
        #     s = r[1] * fps / self.stride
        #     e = r[2] * fps / self.stride
        #     if e < s:
        #         continue
        #     elif e - s < 1.0:
        #         e += 1
        #     s = max(0, int(s))
        #     e = int(e)
        #     active_labels[r[0]].append([s, e, utils.strlist2indlist([r[3]], classlist)[0]])
        # np.save(filepath, np.array(active_labels))
        # return active_labels

    def get_active_labels(self, active_ratio):
        # self.single_labels store the according single frames for each action
        self.single_labels = [[] for _ in range(len(self.segments))]
        for idx in self.trainidx:
            seg = self.segments[idx]
            self.single_labels[idx] = [[] for _ in range(len(seg))]

        single_data = self.read_single_frame_data(os.path.join(self.path_to_annotations, 'single_frames'))
        single_data = single_data[["vid", 'time', 'label']].to_numpy()
        videonames_map = {}
        for i in range(len(self.videoname)):
            videonames_map[self.videoname[i].decode('utf-8')] = i
        for vid, t, c in single_data:
            vid = videonames_map[vid]
            seg = self.segments[vid]
            gtl = self.gtlabels[vid]
            min_dis = 10000
            min_idx = -1
            for idx in range(len(seg)):
                if c != gtl[idx]:
                    continue
                dis = distance_bewteen_an_interval_and_an_point(seg[idx], t)
                if dis < min_dis:
                    min_dis = dis
                    min_idx = idx
            if min_idx == -1:
                print(vid, t, c)
            else:
                self.single_labels[vid][min_idx].append(t)

        self.get_all_actions()
        self.active_labels, self.single_frame_labels = self.random_sampling_actions(active_ratio)
        labels = self.merge_labels()
        return labels
        
    def merge_labels(self):
        labels = []
        for i in range(len(self.videoname)):
            max_len = len(self.features[i])
            frame_label = []
            for j in range(max_len):
                res = self.active_labels[i][j] + self.single_frame_labels[i][j]
                res = list(set(res))
                res.sort()
                frame_label.append(res)
            labels += [frame_label]
        return labels


    def get_all_frame_labels(self):
        classlist = self.get_classlist()
        labels = []
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(max_len)]
            mids = []
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg) * fps / self.stride
                    if end < start:
                        continue
                    elif end - start < 1.0:
                        end += 1
                    start = max(0, int(start))
                    if self.increase:
                        end = min(max_len, int(end + 1))
                    else:
                        end = min(max_len, int(end))
                    for pid in range(start, end):
                        if intl not in frame_label[pid]:
                            frame_label[pid].append(intl)
                labels += [frame_label]
        return labels

    def get_rand_frame_labels(self):
        classlist = self.get_classlist()
        labels = []
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(max_len)]
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg) * fps / self.stride
                    start = max(0, int(np.ceil(start)))
                    if self.increase:
                        end = min(max_len, int(end + 1))
                    else:
                        end = min(max_len, int(end))
                    if end <= start:
                        continue
                    mid = np.random.choice(range(start, end), 1)[0]
                    if intl not in frame_label[mid]:
                        frame_label[mid].append(intl)
                labels += [frame_label]
        return labels

    def load_frame_data(self):
        '''

        load frame data for training

        '''

        features = []
        labels = []
        inds = []
        count_labels = []
        video_labels = []
        frame_labels = []
        cent_labels = []
        frame_ids = []
        classlist = self.get_classlist()
        one_hots = np.eye(len(classlist))

        # random sampling
        rand_sampleid = np.arange(len(self.trainidx))
        np.random.shuffle(rand_sampleid)
        for i in rand_sampleid:
            inds.append(self.trainidx[i])
            idx = self.trainidx[i]
            feat = self.get_feature(idx)
            frame_label = self.get_frame_label(idx)
            #  count_label = np.zeros(len(classlist)+1)
            if len(feat) <= self.t_max:
                feat = np.pad(feat, ((0, self.t_max - len(feat)), (0, 0)),
                              mode='constant')
            else:
                r = np.random.randint(len(feat) - self.t_max)
                feat = feat[r:r + self.t_max]
                frame_label = frame_label[r:r + self.t_max]
            frame_id = [
                i for i in range(len(frame_label)) if len(frame_label[i]) > 0
            ]
            if len(frame_id) < 1:
                continue
            frame_label = [
                np.mean(one_hots[frame_label[i]], axis=0) for i in frame_id
            ]
            count_label = np.sum(frame_label, axis=0)
            video_label = (count_label > 0).astype(np.float32)
            #  video_label[0] = 1.0
            #  count_label[0] = 1
            #  count_label[0] = np.max(count_label)
            video_labels += [video_label]
            count_labels += [count_label]
            frame_labels += [np.array(frame_label)]
            features += [feat]
            frame_ids += [frame_id]
            if len(features) == self.batch_size:
                break

        frame_labels = np.concatenate(frame_labels, 0)
        return np.array(features), np.array(video_labels), np.array(
            count_labels), frame_labels, frame_ids

    def save_self(self, filename):
        pickle_save_data(filename, self)
        return True