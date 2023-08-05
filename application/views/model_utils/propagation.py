# try:
import enum
from SF_Net.utils import graph_propagation, left_bound, right_bound, common_data, construct_csr_matrix_by_edge
from SF_Net.utils.train import anchor_expand
from SF_Net.utils.train.propagation import pair_matching, adaptive_knn
from SF_Net.utils.dataset.video_dataset import left_active, right_active
# except:
#     None
import os
import numpy as np
from scipy.special import softmax
from time import time
import pickle
from copy import deepcopy
import json
from scipy import sparse
from tqdm import tqdm
from scipy import io as sio
from ..utils.helper_utils import left_label_bound, right_label_bound
from ..database_utils.data import encoding_pair, decoding_pair
from threading import Thread
# import drawSvg as draw
from PIL import Image
from ..utils.helper_utils import check_dir


def remove_redundant_keys(knn_pairs):
    new_pairs = {}
    for key in knn_pairs:
        value = knn_pairs[key]
        a, b = key.split("-")
        a = int(a)
        b = int(b)
        if a > b:
            a, b = b, a
        new_key = str(a) + "-" + str(b)
        if new_key not in new_pairs:
            new_pairs[new_key] = value
    return new_pairs

class MatchAndProp:
    def __init__(self, data):
        self.data = data
        self.dataname = self.data.dataname
        self.common_data_root = self.data.common_data_root
        self.step_data_root = self.data.step_data_root
        self.previous_data_root = self.data.previous_data_root
        self.step = self.data.step
        self.debug = self.data.debug

        self.state = 0
        # 0 仅仅更新数据，并不需要将pseudo label送入deep训练
        # 1 更新数据，需要将pseudo label送入deep训练
        self.pred_max = 10

        self.pseudo_label_path = os.path.join(self.step_data_root, "pseudo_label.pkl")
        self.pseudo_labels = None

        self.simi_path = os.path.join(self.step_data_root, "simi_matrix.npy")
        self.simi_matrix = None

        self.constraints_path = os.path.join(self.step_data_root, "constraints.json")
        self.constraints = {}
        self.full_constraints = {}

        self.label_constraints = {}

        self.classification_prediction_path = os.path.join(self.step_data_root, "classification_prediction.npy")
        self.classification_prediction = None

        self.action_prediction_path = os.path.join(self.step_data_root, "action_prediction.npy")
        self.action_prediction = None

        self.propagation_prediction_path = os.path.join(self.step_data_root, "propagation.npy")
        self.propagation_prediction = None

        self.affinity_matrix_path = os.path.join(self.step_data_root, "affinity_matrix.mtx")
        self.affinity_matrix = None

        self.all_pairs_path = os.path.join(self.step_data_root, "all_pairs.json")
        self.all_pairs_dict = None

        self.knn_pairs_path = os.path.join(self.step_data_root, "knn_pairs.json")
        self.knn_pairs_dict = None
        self.knn_pairs_map = None

        self.changed_pairs = []

        self.active_ids = []

        self.old_pseudo_action = None

        self.parameters = {
            'alpha_1': 0.5,
            'alpha_2': 1,
            'beta': 1.0,
            'pv': 0.90,
            'A': 1,
            'radious': 3,
            'special_alpha': 4,
        }

        self.history = {
            'boundary': [],
            'must_link': [],
            'must_not_link': [],
            'pseudo_action': [],
        }

    def run(self):
        self.init()
        self.update(self.constraints.copy())
        # if self.state == 1:
        self.save_state(self.step_data_root)
            # print('******************** deep model training starts *****************************')
            # t = Thread(target=main, args=(self.train_parameters, )) 
            # t.start()

    def load_prerun_result(self):
        self.load_state(self.step_data_root)

    def get_old_pseudo_action(self):
        self.load_prerun_result()
        self.old_pseudo_action = deepcopy(self.get_all_pseudo_actions())

    def get_data_for_next(self):
        self.history['boundary'].append(len(self.active_ids))
        must_match = 0
        must_not_match = 0
        for key in self.full_constraints.keys():
            constraint = self.full_constraints[key]
            if 'must_match' in constraint.keys():
                must_match += len(constraint['must_match'])
            if 'must_not_match' in constraint.keys():
                must_not_match += len(constraint['must_not_match'][0])
                must_not_match += len(constraint['must_not_match'][1])

        self.history['must_link'].append(must_match)
        self.history['must_not_link'].append(must_not_match)
        self.history['pseudo_action'].append(self.old_pseudo_action)
        save_data = {
            'cls_pred': deepcopy(self.classification_prediction),
            'act_pred': deepcopy(self.action_prediction),
            'knn_pairs': deepcopy(self.knn_pairs_dict),
            'simi_matrix': deepcopy(self.simi_matrix),
            'pseudo_labels': deepcopy(self.pseudo_labels),
            'propagation_prediction': deepcopy(self.propagation_prediction),
            'active_ids': deepcopy(self.active_ids),
            'history': self.history,
        }
        return save_data

    def save_data_for_next(self, save_data):
        np.save(self.classification_prediction_path, save_data['cls_pred'])
        np.save(self.action_prediction_path, save_data['act_pred'])
        np.save(self.simi_path, save_data['simi_matrix'])
        self.pseudo_labels = save_data['pseudo_labels']
        with open(self.pseudo_label_path, 'wb') as f:
            pickle.dump(save_data['pseudo_labels'], f)
        self.propagation_prediction = save_data['propagation_prediction']
        np.save(self.propagation_prediction_path, save_data['propagation_prediction'])
        with open(self.knn_pairs_path, 'w') as f:
            f.write(json.dumps(save_data['knn_pairs']))
        np.save(os.path.join(self.step_data_root, 'active_ids.npy'), save_data['active_ids'])
        self.active_ids = list(set(self.active_ids + save_data['active_ids']))
        self.simi_matrix = save_data['simi_matrix']
        self.save_action_simi_matrix(self.step_data_root)
        self.history = save_data['history']
        with open(os.path.join(self.step_data_root, 'history.json'), 'w') as f:
            f.write(json.dumps(self.history))

    def add_active(self, active_ids):
        self.active_ids = list(set(self.active_ids + deepcopy(active_ids)))

    def update(self, constraints):
        # based on deep model outputs (action and classification prediction) from self.data
        # Otherwise, incrementally update the alignment results and propagation.

        alpha_1 = self.parameters['alpha_1']
        alpha_2 = self.parameters['alpha_2']
        beta = self.parameters['beta']
        pv = self.parameters['pv']
        A = self.parameters['A']
        radious = self.parameters['radious']

        # update classification prediction by boundary
        self.update_cls_prediction_by_boundary()
        # update classification prediction by label
        self.update_cls_prediction_by_label()

        logit = self.propagation_base_on_graph(A, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta, constraints=self.constraints)
        self.propagation_prediction = logit

        start = 0

        new_pseudo_label = deepcopy(self.data.single_frame_labels)
        for idx in self.data.train_vid:
            length = len(self.data.single_frame_labels[idx])
            cur_label = deepcopy(self.data.single_frame_labels[idx])
            for active_id in self.active_ids:
                videoid = int(active_id.split('-')[0])
                frameid = int(active_id.split('-')[1])
                if videoid == idx:
                    cur_label[frameid] = []
                    # active label doesn't paticipate in expanding
            new_label = anchor_expand(logit[start: start + length][:, 1:], cur_label, None, pv=pv, radious=radious)
            start += length
            for i in range(len(cur_label)):
                res = new_label[i] + self.data.single_frame_labels[idx][i]
                res = list(set(res))
                res.sort()
                new_label[i] = res
            new_pseudo_label[idx] = new_label

        # update pseudo labels by boundary
        self.update_pseudo_labels_by_boundary(new_pseudo_label)
        self.pseudo_labels = new_pseudo_label

        # self.save_action_simi_matrix(self.step_data_root)

    def save_state(self, dir):
        # save constraints, pairs, pseudo labels, propagation prediction and affinity matrix
        return 
        # abandoned


        

        # path = os.path.join(dir, "constraints.json")
        # with open(path, 'w') as f:
        #     f.write(json.dumps(self.full_constraints))

        # path = os.path.join(dir, "all_pairs.json")
        # with open(path, 'w') as f:
        #     f.write(json.dumps(self.all_pairs_dict))

        # path = os.path.join(dir, "knn_pairs.json")
        # with open(path, 'w') as f:
        #     f.write(json.dumps(self.knn_pairs_dict))

        # path = os.path.join(dir, "pseudo_label.pkl")
        # with open(path, 'wb') as f:
        #     pickle.dump(self.pseudo_labels, f)

        # path = os.path.join(dir, "propagation.npy")
        # np.save(path, self.propagation_prediction)

        # path = os.path.join(dir, "simi_matrix.npy")
        # np.save(path, self.simi_matrix)

        # path = os.path.join(dir, "affinity_matrix.mtx")
        # sio.mmwrite(path, self.affinity_matrix)

        # self.save_action_simi_matrix(dir)

        print("******************* saved *******************")

    def load_state(self, dir):
        # load constraints, pairs, pseudo labels and propagation prediction
        # path = os.path.join(dir, "constraints.json")
        # with open(path, 'r') as f:
        #     self.full_constraints = json.load(f)

        # path = os.path.join(dir, "all_pairs.json")
        # with open(path, 'r') as f:
        #     self.all_pairs_dict = json.load(f)

        # path = os.path.join(dir, "knn_pairs.json")
        # with open(path, 'r') as f:
        #     self.knn_pairs_dict = json.load(f)            
        
        path = os.path.join(dir, "pseudo_label.pkl")
        with open(path, 'rb') as f:
            self.pseudo_labels = pickle.load(f)

        path = os.path.join(dir, "propagation.npy")
        self.propagation_prediction = np.load(path)

        path = os.path.join(dir, "simi_matrix.npy")
        self.simi_matrix = np.load(path)

        path = os.path.join(dir, "active_ids.npy")
        if os.path.exists(path):
            self.active_ids = np.load(path)
            self.active_ids = list(self.active_ids)

        path = os.path.join(dir, 'history.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.history = json.load(f)

        print("******************* loaded ******************")

    def update_knn_pairs_map(self):
        self.knn_pairs_dict = remove_redundant_keys(self.knn_pairs_dict)
        self.knn_pairs_map = {}
        for key in self.knn_pairs_dict:
            a, b = decoding_pair(key)
            if a not in self.knn_pairs_map:
                self.knn_pairs_map[a] = []
            if b not in self.knn_pairs_map:
                self.knn_pairs_map[b] = []
            self.knn_pairs_map[a].append(key)
            self.knn_pairs_map[b].append(key)

    def get_knn_pairs(self):
        if self.knn_pairs_dict is None:
            with open(self.knn_pairs_path, 'r') as f:
                self.knn_pairs_dict = json.load(f)
            self.update_knn_pairs_map()

        return self.knn_pairs_dict
    
    def get_knn_pair(self, idx):
        if self.knn_pairs_map is None:
            self.get_knn_pairs()
        return self.knn_pairs_map[idx]
    
    def get_all_pairs(self):
        if self.all_pairs_dict is None:
            with open(self.all_pairs_path, 'r') as f:
                self.all_pairs_dict = json.load(f)
        return self.all_pairs_dict

    def get_pseudo_labels(self):
        return deepcopy(self.pseudo_labels)

    def get_propatation_results(self):
        return deepcopy(self.propagation_prediction)

    def get_all_pseudo_actions(self):
        # self.pseudo_labels = self.data.gt_labels
        thres = 0.5
        inf = 1000000
        ln = len(self.data.single_frames)
        boundary = self.convert_active_to_boundary(self.active_ids)
        for single_index, single_frame in enumerate(self.data.single_frames):
            video_id, frame_id = decoding_pair(single_frame["id"])
            single_frame["video_id"] = video_id
            single_frame["single_frame"] = frame_id
            label = single_frame["label"][0]
            # get selected video prediction
            bound = self.get_video_bound_idx(video_id)
            prediction = self.propagation_prediction[bound[0]: bound[1]]
            # judge neighbour pos
            left_limit, right_limit = -1, inf
            if single_index > 0:
                pre_single_frame = self.data.single_frames[single_index - 1]
                pre_video_id, pre_frame_id = decoding_pair(pre_single_frame["id"])
                if pre_video_id == video_id:
                    left_limit = pre_frame_id
            if single_index < ln - 1:
                post_single_frame = self.data.single_frames[single_index + 1]
                post_video_id, post_frame_id = decoding_pair(post_single_frame["id"])
                if post_video_id == video_id:
                    right_limit = post_frame_id

            # get bound of single frame action
            pseudobound1, predboundl = left_label_bound(self.pseudo_labels[video_id], \
                frame_id, label, prediction, thres, left_limit)
            pseudoboundr, predboundr = right_label_bound(self.pseudo_labels[video_id], \
                frame_id, label, prediction, thres, right_limit)
                
            for bound in boundary:
                if single_index == bound['idx']:
                    predboundl = bound['lgt']
                    predboundr = bound['rgt']
                    pseudobound1 = bound['lgt']
                    pseudoboundr = bound['rgt']
                    break

            single_frame["left_bound"] = predboundl
            single_frame["right_bound"] = predboundr
            single_frame["left_pseudo"] = pseudobound1
            single_frame["right_pseudo"] = pseudoboundr
            # single_frame["single_index"] = single_index
            if single_frame['id'] in self.active_ids:
                single_frame["left_bound"] = single_frame["left_pseudo"]
                single_frame["right_bound"] = single_frame["right_pseudo"]
                predboundl = single_frame["left_pseudo"]
                predboundr = single_frame["right_pseudo"]
            if predboundr == frame_id - 1:
                predboundr = frame_id
                single_frame["right_bound"] = frame_id
            pred_scores = []
            for i in range(predboundl, frame_id):
                pred_scores.append(round(prediction[i][label+1], 5))
            pred_scores.append(1)
            for i in range(frame_id + 1, predboundr + 1):
                pred_scores.append(round(prediction[i][label+1], 5))
            single_frame["pred_scores"] = pred_scores
        return self.data.single_frames.copy()

    def get_history(self):
        return deepcopy(self.history)

    def get_history_action(self, step):
        assert len(self.history['pseudo_action']) > step
        return self.history['pseudo_action'][step]
    
    def get_history_action_length(self, step):
        assert len(self.history['pseudo_action']) > step
        action = self.history['pseudo_action'][step]
        lengths = {}
        for item in action:
            id = item['id']
            lb = item['left_bound']
            rb = item['right_bound']
            lengths[id] = rb - lb + 1
        return lengths

    def get_history_action_length_by_class(self, step, class_num):
        assert len(self.history['pseudo_action']) > step
        action = self.history['pseudo_action'][step]
        lengths = {}
        for item in action:
            id = item['id']
            if class_num in item['label']:
                lb = item['left_bound']
                rb = item['right_bound']
                lengths[id] = rb - lb + 1
        return lengths

    def get_history_info(self):
        return {
            'boundary': self.history['boundary'],
            'must_link': self.history['must_link'],
            'must_not_link': self.history['must_not_link'],
        }

    def get_pred_scores_of_video_with_given_boundary(self, video_id, label, start, end):
        bound = self.get_video_bound_idx(video_id)
        length = bound[1] - bound[0]
        prediction = self.propagation_prediction[bound[0]: bound[1]]
        if start < 0:
            start = 0
        if end >= length:
            end = length - 1
        assert start <= end
        pred_scores = []
        for i in range(start, end + 1):
            pred_scores.append(round(prediction[i][label+1], 5))
        return [start, end], pred_scores 

    def set_pred_scores_of_video_with_given_boundary(self, video_id, label, start, end, scores):
        bound = self.get_video_bound_idx(video_id)
        for i in range(end - start + 1):
            self.propagation_prediction[bound[0] + i + start][label+1] = scores[i]

    def get_single_frame_by_single_idx(self, idx):
        return self.data.get_single_frame_by_single_idx(idx)

    def get_affinity_matrix(self):
        if self.affinity_matrix is None:
            self.affinity_matrix = sio.mmread(os.path.join(self.step_data_root, "affinity_matrix.mtx"))
        res = self.affinity_matrix.copy()
        res = sparse.csr_matrix(res)
        return res

    def add_pairs_in_knn_pairs(self, id):
        paths = self.step_data_root.split('/')
        paths[-1] = 'step0'
        paths = paths[1:]
        path = ''
        for it in paths:
            path += '/' + it
        with open(path + '/all_pairs.json', 'r') as f:
            pairs = json.load(f)
        pair = pairs[id]
        self.get_knn_pairs()
        self.knn_pairs_dict[id] = pair
        self.update_knn_pairs_map()

    def get_video_bound_idx(self, v_id):
        idx0 = self.data.get_idx_by_video_and_frame_id(v_id, 0)
        length = len(self.pseudo_labels[v_id])
        return [idx0, idx0 + length]

    def get_pred_score_of_video(self, vid, action_class):
        # return a list of pred scores
        res = []
        labels = self.pseudo_labels[vid]
        for label in labels:
            if action_class in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def propagation_base_on_graph(self, A, alpha_1=0.5, alpha_2=1, beta=0, constraints={}):
        all_features = self.data.get_all_features()
        whole_to_invdual = self.data.whole_to_indvdual
        idvdual_to_whole = self.data.indvdual_to_whole
        all_preds = self.classification_prediction
        labeled_frames = self.data.single_frames

        # if self.step == 0:
        #     self.generate_knn_pairs_dict()
        confident_unlabeled, special_labeled, deleted = self.update_pairs_and_similarity(all_features, idvdual_to_whole, labeled_frames, all_preds, A, constraints)
        # update pairs and similarity by constraints
        # confident_unlabeled = []
        # self.update_special_labeled(special_labeled)

        self.affinity_matrix = self.generate_knn_affinity_matrix(deleted)
        # construct graph by pairs and similarity

        class_num = len(self.data.classlist)

        tau_changed = []

        inds = self.data.train_vid
        start_id = 0
        labeled_start_id = 0
        center_map = []
        for idx in inds:
            labeled_idx = []
            num = len(self.data.gt_labels[idx])
            end = start_id + num
            while labeled_start_id < len(labeled_frames) and start_id <= labeled_frames[labeled_start_id]['idx'] < end:
                if labeled_frames[labeled_start_id]['single_idx'] in self.label_constraints.keys():
                    label = self.label_constraints[labeled_frames[labeled_start_id]['single_idx']]['label']
                    for i in range(label[0], label[1] + 1):
                        frameid = labeled_frames[labeled_start_id]['id'].split('-')[0] + '-' + str(i)
                        labeled_idx.append(idvdual_to_whole[frameid])
                    
                    fid = int(labeled_frames[labeled_start_id]['id'].split('-')[1])
                    if label[0] < fid:
                        for i in range(label[0] - 3, label[0]):
                            frameid = labeled_frames[labeled_start_id]['id'].split('-')[0] + '-' + str(i)
                            tau_changed.append(idvdual_to_whole[frameid])
                    if label[1] > fid:
                        for i in range(label[1] + 1, label[1] + 4):
                            frameid = labeled_frames[labeled_start_id]['id'].split('-')[0] + '-' + str(i)
                            tau_changed.append(idvdual_to_whole[frameid])
                else:
                    labeled_idx.append(labeled_frames[labeled_start_id]['idx'])
                labeled_start_id += 1

            if len(labeled_idx) == 0:
                index = np.ones(num) * -1
                center_map.append(index)
            else:
                labeled_idx = np.array(labeled_idx)
                index = np.arange(start_id, end)
                index = np.repeat(index, labeled_idx.shape[0]).reshape(end - start_id, labeled_idx.shape[0])
                index = np.abs(index - labeled_idx)
                index = np.argmin(index, axis=1)
                index = labeled_idx[index]
                center_map.append(index)
            start_id = end

        center_map = np.concatenate(center_map)
        center_map = center_map.astype(int)

        scores = np.array(all_preds, dtype='float64')
        label_distributions, unnorm_dist = graph_propagation(self.affinity_matrix, scores, center_map, 
            normalized=True, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta, max_iter=20, confident_unlabeled=confident_unlabeled,
            special_labeled=special_labeled, tau_changed=tau_changed)
        return label_distributions

    def generate_knn_pairs_dict(self):
        all_features = self.data.get_all_features()
        all_preds = self.classification_prediction
        labeled_frames = self.data.single_frames
        A = self.parameters['A']
        # generate knn affinity matrix according to pairs
        labeled_features = [i["idx"] for i in labeled_frames]
        labeled_preds = all_preds[np.array(labeled_features)]
        labeled_features = all_features[np.array(labeled_features)]

        kmax = 7

        knn_pairs = {}
        for count_idx, labeled_frame in tqdm(enumerate(labeled_frames)):
            test_feature = labeled_features[count_idx]
            topk = adaptive_knn(labeled_features, labeled_preds, test_feature, kmax, A)
            simi = self.simi_matrix[count_idx]
            sorted_idx = simi.argsort()[::-1]
            labels = []
            for idx in sorted_idx:
                if count_idx == idx:
                    continue
                # if idx == 1610 and count_idx != 1610:
                #     continue
                # elif count_idx == 1610 and idx in [1610, 1609, 1611, 1608]:
                #     continue
                # if count_idx == 1610:
                #     topk = 3
                key = str(count_idx) + '-' + str(idx)
                if key not in self.all_pairs_dict.keys():
                    continue
                pairs = self.all_pairs_dict[key]
                knn_pairs[key] = pairs
                pairs = [(p[0], p[1]) for p in pairs]
                labels.append(idx)
                if len(labels) >= topk:
                    break
        print("******************* final *******************")
        self.knn_pairs_dict = knn_pairs
        self.knn_pairs_dict = remove_redundant_keys(self.knn_pairs_dict)

    def generate_knn_affinity_matrix(self, deleted):
        # generate knn affinity matrix according to pairs
        all_features = self.data.get_all_features()
        edges= []

        for key in self.knn_pairs_dict.keys():
            pairs = self.knn_pairs_dict[key]
            pairs = [(p[0], p[1]) for p in pairs if p[0] not in deleted and p[1] not in deleted]
            edges += pairs
        for item in self.changed_pairs:
            edges += item
        # print("******************* final *******************")
        return construct_csr_matrix_by_edge(edges, all_features.shape[0])

    def update_pairs_and_similarity(self, all_features, idvdual_to_whole, labeled_frames, all_preds, A, constraints):
        single_frame_labels = self.data.single_frame_labels
        indvdual_to_whole = self.data.indvdual_to_whole
        for labeled_frame in labeled_frames:
            idx = labeled_frame['idx']
            videoid = int(labeled_frame['id'].split('-')[0])
            frameid = int(labeled_frame['id'].split('-')[1])
            frames = single_frame_labels[videoid]
            labeled_frame['left'] = left_bound(frames, frameid, frames[frameid][0])
            labeled_frame['right'] = right_bound(frames, frameid, frames[frameid][0])
            labeled_frame['active'] = False

        kmax = 7

        confident_unlabeled = []
        special_labeled = []
        deleted = []

        for key in constraints.keys():
            constraint = constraints[key]
            idx = int(key.split('-')[0])
            idy = int(key.split('-')[1])

            if idx > idy:
                temp = idx
                idx = idy
                idy = temp

            labeled_framex = labeled_frames[idx]
            labeled_framey = labeled_frames[idy]

            videoidx = int(labeled_framex['id'].split('-')[0])
            startx = labeled_framex['left']
            endx = labeled_framex['right']
            labeled_pairx = {
                "lb": startx,
                "rb": endx,
                "v_id": videoidx
            }

            videoidy = int(labeled_framey['id'].split('-')[0])
            starty = labeled_framey['left']
            endy = labeled_framey['right']
            labeled_pairy = {
                "lb": starty,
                "rb": endy,
                "v_id": videoidy
            }
            print(idx, idy, labeled_pairx, labeled_pairy)
            print(constraint)
            near = False
            if (idx - idy == 1 or idy - idx == 1) and videoidx == videoidy and not labeled_framey['active'] and not\
            labeled_framex['active']:
                near = True
            pairs, cost = pair_matching(all_features, all_preds, idvdual_to_whole, labeled_pairx, labeled_pairy, kmax, A, near, deepcopy(constraint))

            for key in self.label_constraints.keys():
                lb = self.label_constraints[key]['label'][0]
                rb = self.label_constraints[key]['label'][1]
                if key == idx:
                    single_frame = self.data.single_frames[idx]
                    frame_id = int(single_frame['id'].split('-')[1])
                    video_id = int(single_frame['id'].split('-')[0])
                    if lb < frame_id:
                        current = lb - 4
                        while True:
                            begin = [p[0] for p in pairs]
                            if str(video_id) + '-' + str(current) not in begin:
                                break
                            pairs = [p for p in pairs if p[0] != str(video_id) + '-' + str(current)]
                            deleted.append(idvdual_to_whole[str(video_id) + '-' + str(current)])
                            current -= 1
                    elif rb > frame_id:
                        current = rb + 4
                        while True:
                            begin = [p[0] for p in pairs]
                            if str(video_id) + '-' + str(current) not in begin:
                                break
                            pairs = [p for p in pairs if p[0] != str(video_id) + '-' + str(current)]
                            deleted.append(idvdual_to_whole[str(video_id) + '-' + str(current)])
                            current += 1
                elif key == idy:
                    single_frame = self.data.single_frames[idy]
                    frame_id = int(single_frame['id'].split('-')[1])
                    video_id = int(single_frame['id'].split('-')[0])
                    if lb < frame_id:
                        current = lb - 4
                        while True:
                            end = [p[1] for p in pairs]
                            if str(video_id) + '-' + str(current) not in begin:
                                break
                            pairs = [p for p in pairs if p[1] != str(video_id) + '-' + str(current)]
                            deleted.append(idvdual_to_whole[str(video_id) + '-' + str(current)])
                            current -= 1
                    elif rb > frame_id:
                        current = rb + 4
                        while True:
                            begin = [p[1] for p in pairs]
                            if str(video_id) + '-' + str(current) not in begin:
                                break
                            pairs = [p for p in pairs if p[1] != str(video_id) + '-' + str(current)]
                            deleted.append(idvdual_to_whole[str(video_id) + '-' + str(current)])
                            current += 1
            # delete background edges and record information
            
            pairs = [(idvdual_to_whole[p[0]], idvdual_to_whole[p[1]]) for p in pairs]
            if len(pairs) == 0:
                continue

            # if self.step == 0:
            #     self.all_pairs_dict[str(idx) + '-' + str(idy)] = pairs
            #     self.all_pairs_dict[str(idy) + '-' + str(idx)] = pairs

            if self.knn_pairs_dict is not None:
                if str(idx) + '-' + str(idy) in self.knn_pairs_dict.keys():
                    self.knn_pairs_dict[str(idx) + '-' + str(idy)] = pairs
                if str(idy) + '-' + str(idx) in self.knn_pairs_dict.keys():
                    self.knn_pairs_dict[str(idy) + '-' + str(idx)] = pairs

            self.simi_matrix[idx][idy] = cost / len(pairs)
            self.simi_matrix[idy][idx] = cost / len(pairs)
            self.changed_pairs.append([[p[1], p[0]] for p in pairs])

            for p in pairs:
                special_labeled.append(p[0])
                special_labeled.append(p[1])

            if 'boundary' in constraint.keys():
                boundary = constraint['boundary']
                if len(boundary[0]) == 2 and len(boundary[1]) == 2:
                    continue
                elif len(boundary[0]) == 2:
                    lb = boundary[0][0]
                    rb = boundary[0][1]
                    lb = indvdual_to_whole[str(videoidx) + '-' + str(lb)]
                    rb = indvdual_to_whole[str(videoidx) + '-' + str(rb)]
                    for p in pairs:
                        if lb <= p[0] <= rb:
                            confident_unlabeled.append(p[1])
                elif len(boundary[1]) == 2:
                    lb = boundary[1][0]
                    rb = boundary[1][1]
                    lb = indvdual_to_whole[str(videoidy) + '-' + str(lb)]
                    rb = indvdual_to_whole[str(videoidy) + '-' + str(rb)]
                    for p in pairs:
                        if lb <= p[1] <= rb:
                            confident_unlabeled.append(p[0])

            if 'must_not_match' in constraint.keys():
                must_not_match = constraint['must_not_match']
                for item in must_not_match[0]:
                    index = indvdual_to_whole[str(videoidx) + '-' + str(item)]
                    confident_unlabeled.append(index)
                for item in must_not_match[1]:
                    index = indvdual_to_whole[str(videoidy) + '-' + str(item)]
                    confident_unlabeled.append(index)

        confident_unlabeled = list(set(confident_unlabeled))
        confident_unlabeled.sort()

        special_labeled = list(set(special_labeled))
        special_labeled.sort()

        deleted = list(set(deleted))
        deleted.sort()
        # special_labeled = None
        return confident_unlabeled, special_labeled, deleted

    def get_pairs_and_similarity(self, A):
        all_features = self.data.get_all_features()
        whole_to_invdual = self.data.whole_to_indvdual
        idvdual_to_whole = self.data.indvdual_to_whole
        all_preds = self.classification_prediction
        labeled_frames = self.data.single_frames
        simi = np.zeros((len(labeled_frames), len(labeled_frames)))
        pairs_dict = {}
        kmax = 7

        frame_labels = self.data.single_frame_labels
        for labeled_frame in labeled_frames:
            idx = labeled_frame['idx']
            videoid = int(labeled_frame['id'].split('-')[0])
            frameid = int(labeled_frame['id'].split('-')[1])
            frames = frame_labels[videoid]
            labeled_frame['left'] = left_bound(frames, frameid, frames[frameid][0])
            labeled_frame['right'] = right_bound(frames, frameid, frames[frameid][0])
            labeled_frame['active'] = False

        old_pairx = {}
        old_pairy = {}
        old_pairs = None
        old_cost = 0

        for i, labeled_framex in enumerate(labeled_frames):
            for j, labeled_framey in enumerate(labeled_frames):
                if i > j:
                    continue
                if not common_data(labeled_framex['label'], labeled_framey['label']):
                    continue
                videoidx = int(labeled_framex['id'].split('-')[0])
                startx = labeled_framex['left']
                endx = labeled_framex['right']
                labeled_pairx = {
                    "lb": startx,
                    "rb": endx,
                    "v_id": videoidx
                }

                videoidy = int(labeled_framey['id'].split('-')[0])
                starty = labeled_framey['left']
                endy = labeled_framey['right']
                labeled_pairy = {
                    "lb": starty,
                    "rb": endy,
                    "v_id": videoidy
                }
                print(i, j, labeled_pairx, labeled_pairy)
                near = False
                if j - i == 1 and videoidx == videoidy and not labeled_framey['active'] and not\
                labeled_framex['active']:
                    near = True
                if old_pairx == labeled_pairx and old_pairy == labeled_pairy:
                    pairs = old_pairs
                    cost = old_cost
                else:
                    pairs, cost = pair_matching(all_features, all_preds, idvdual_to_whole, labeled_pairx, labeled_pairy, kmax, A, near)
                    pairs = [(idvdual_to_whole[p[0]], idvdual_to_whole[p[1]]) for p in pairs]
                if len(pairs) == 0:
                    continue
                simi[i][j] = cost / len(pairs)
                simi[j][i] = cost / len(pairs)

                pairs_dict[str(i) + '-' + str(j)] = pairs
                pairs_dict[str(j) + '-' + str(i)] = pairs

                old_pairx = labeled_pairx
                old_pairy = labeled_pairy
                old_pairs = pairs
                old_cost = cost

        self.all_pairs_dict = pairs_dict
        self.simi_matrix = simi

    def generate_full_affinity_matrix(self):
        # generate full affinity matrix according to pairs
        all_features = self.data.get_all_features()
        edges= []

        for key in self.all_pairs_dict.keys():
            pairs = self.all_pairs_dict[key]
            pairs = [(p[0], p[1]) for p in pairs]
            edges += pairs
        print("******************* final *******************")
        return construct_csr_matrix_by_edge(edges, all_features.shape[0])

    def save_action_simi_matrix(self, dir):
        single_frames = self.data.single_frames
        classlist = self.data.classlist
        action_ids = [[] for _ in classlist]
        for idx, single_frame in enumerate(single_frames):
            labels = single_frame['label']
            for label in labels:
                action_ids[label].append(idx)
        for i, name in enumerate(classlist):
            action_id = action_ids[i]
            simi = np.zeros((len(action_id), len(action_id)))
            for j, action_id_j in enumerate(action_id):
                for k, action_id_k in enumerate(action_id):
                    simi[j][k] = self.simi_matrix[action_id_j][action_id_k]
            path = os.path.join(dir, 'cluster/similarity/' + str(i) + '.npy')
            np.save(path, simi)

    def init(self):
        assert os.path.exists(self.classification_prediction_path)
        assert os.path.exists(self.action_prediction_path)

        if self.classification_prediction is None:
            self.classification_prediction = np.load(self.classification_prediction_path)
            if self.step == 0:
                self.classification_prediction = softmax(self.classification_prediction, axis=1)
                for labeled_frame in self.data.single_frames:
                    idx = int(labeled_frame['idx'])
                    labels = labeled_frame['label']
                    pred = np.zeros(self.classification_prediction.shape[1])
                    pred[labels[0] + 1] = 1
                    self.classification_prediction[idx] = pred

        if self.action_prediction is None:
            self.action_prediction = np.load(self.action_prediction_path)

        # if self.step == 0:
        #     if self.all_pairs_dict is None or self.simi_matrix is None:
        #         initial_pairs_path = os.path.join(self.step_data_root, 'all_pairs.json')
        #         initial_simi_matrix_path = os.path.join(self.step_data_root, 'simi_matrix.npy')
        #         if os.path.exists(initial_pairs_path) and os.path.exists(initial_simi_matrix_path):
        #             with open(initial_pairs_path, 'r') as f:
        #                 self.all_pairs_dict = json.load(f)
        #             self.simi_matrix = np.load(initial_simi_matrix_path)
        #         else:
        #             print('buffer not exists, start to generate pairs and simi matrix')
        #             self.get_pairs_and_similarity(self.parameters['A'])
        # else:
            # if self.all_pairs_dict is None:
            #     prefix = os.path.join(self.data.common_data_root, "step" + str(self.step - 1))
            #     all_pairs_path = os.path.join(prefix, 'all_pairs.json')
            #     with open(all_pairs_path, 'r') as f:
            #         self.all_pairs_dict = json.load(f)

            # if self.knn_pairs_dict is None:
            #     knn_pairs_path = os.path.join(prefix, 'knn_pairs.json')
            #     with open(knn_pairs_path, 'r') as f:
            #         self.knn_pairs_dict = json.load(f)
            # prefix = os.path.join(self.data.common_data_root, "step" + str(self.step - 1))
        prefix = self.step_data_root
        self.get_knn_pairs()

        if self.simi_matrix is None:
            simi_matrix_path = os.path.join(prefix, 'simi_matrix.npy')
            self.simi_matrix = np.load(simi_matrix_path)
        print('********************** init *********************')

    def add_constraint(self, constraints):
        self.constraints = constraints
        for key in self.constraints:
            self.full_constraints[key] = self.constraints[key]

    def add_label_constraint(self, constraints):
        self.label_constraints = deepcopy(constraints)

    def set_state_final(self):
        self.state = 1
    
    def set_state_start(self):
        self.state = 0

    def update_train_parameters(self, parameters):
        self.train_parameters = parameters

    def get_symmetrical_affinity_matrix(self):
        # if self.affinity_matrix is None:
        self.get_knn_pairs()
        res = self.generate_knn_affinity_matrix([])
        res = sparse.csr_matrix(res)
        return res
    
    def convert_active_to_boundary(self, active_ids):
        boundary = []
        single_frames = self.data.single_frames
        for aid in active_ids:
            for idx, single_frame in enumerate(single_frames):
                if single_frame["id"] == aid:
                    boundary.append({
                        "idx": idx,
                        "lgt": single_frame['lgt'][0],
                        "rgt": single_frame['rgt'][0]
                    })
                    break
        return boundary

    def convert_active_to_constraints(self, active_ids):
        self.get_knn_pairs()
        single_frame_labels = self.data.single_frame_labels
        gt_labels = self.data.gt_labels

        # get active_action_id
        active_action_id = []
        single_frames = self.data.single_frames # TODO: use a dict to accelerate
        for aid in active_ids:
            for idx, single_frame in enumerate(single_frames):
                if single_frame["id"] == aid:
                    active_action_id.append({
                        "idx": idx,
                        "lgt": single_frame['lgt'][0],
                        "rgt": single_frame['rgt'][0]
                    })
                    break
        
        # convert it to constraints
        constraints = {}
        for aid in active_action_id:
            neighbors = self.knn_pairs_map[aid['idx']]
            for nei in neighbors:
                a, b = decoding_pair(nei)
                if a == aid["idx"]:
                    constraints[nei] = {'boundary': [[aid["lgt"], aid["rgt"]], []]}
                elif b == aid["idx"]:
                    constraints[nei] = {'boundary': [[], [aid["lgt"], aid["rgt"]]]}
                else:
                    raise ValueError("error")

        return constraints

    def update_cls_prediction_by_boundary(self):
        indvdual_to_whole = self.data.indvdual_to_whole
        for key in self.constraints.keys():
            [a, b] = key.split('-')
            a = int(a)
            b = int(b)
            idx = min(a, b)
            idy = max(a, b)
            if 'boundary' in self.constraints[key].keys():
                xbound = self.constraints[key]['boundary'][0]
                ybound = self.constraints[key]['boundary'][1]
                if len(xbound) == 2:
                    videoid = self.data.single_frames[idx]['id']
                    videoid = int(videoid.split('-')[0])
                    label = self.data.single_frames[idx]['label'][0]
                    for j in range(xbound[0], xbound[1] + 1):
                        frameid = str(videoid) + '-' + str(j)
                        pred = np.zeros(self.classification_prediction.shape[1])
                        pred[label + 1] = self.pred_max
                        self.classification_prediction[indvdual_to_whole[frameid]] = pred
                if len(ybound) == 2:
                    videoid = self.data.single_frames[idy]['id']
                    videoid = int(videoid.split('-')[0])
                    label = self.data.single_frames[idy]['label'][0]
                    for j in range(ybound[0], ybound[1] + 1):
                        frameid = str(videoid) + '-' + str(j)
                        pred = np.zeros(self.classification_prediction.shape[1])
                        pred[label + 1] = self.pred_max
                        self.classification_prediction[indvdual_to_whole[frameid]] = pred

    def update_cls_prediction_by_label(self):
        indvdual_to_whole = self.data.indvdual_to_whole
        for key in self.label_constraints.keys():
            idx = key
            if 'label' in self.label_constraints[key].keys():
                lb = self.label_constraints[key]['label'][0]
                rb = self.label_constraints[key]['label'][1]
                videoid = self.data.single_frames[idx]['id']
                videoid = int(videoid.split('-')[0])
                label = self.data.single_frames[idx]['label'][0]
                for j in range(lb, rb + 1):
                    frameid = str(videoid) + '-' + str(j)
                    pred = np.zeros(self.classification_prediction.shape[1])
                    pred[label + 1] = self.pred_max
                    self.classification_prediction[indvdual_to_whole[frameid]] = pred

    def update_special_labeled(self, special_labeled):
        indvdual_to_whole = self.data.indvdual_to_whole
        for key in self.label_constraints.keys():
            idx = key
            if 'label' in self.label_constraints[key].keys():
                lb = self.label_constraints[key]['label'][0]
                rb = self.label_constraints[key]['label'][1]
                videoid = self.data.single_frames[idx]['id']
                videoid = int(videoid.split('-')[0])
                label = self.data.single_frames[idx]['label'][0]
                for j in range(lb, rb + 1):
                    frameid = str(videoid) + '-' + str(j)
                    special_labeled.append(indvdual_to_whole[frameid])

    def update_pseudo_labels_by_boundary(self, new_pseudo_label):
        for key in self.full_constraints.keys():
            [a, b] = key.split('-')
            a = int(a)
            b = int(b)
            idx = min(a, b)
            idy = max(a, b)
            if 'boundary' in self.full_constraints[key].keys():
                xbound = self.full_constraints[key]['boundary'][0]
                ybound = self.full_constraints[key]['boundary'][1]
                if len(xbound) == 2:
                    videoid = self.data.single_frames[idx]['id']
                    videoid = int(videoid.split('-')[0])
                    frames = new_pseudo_label[videoid]
                    label = self.data.single_frames[idx]['label'][0]
                    for j in range(xbound[0], xbound[1] + 1):
                        if label not in frames[j]:
                            frames[j].append(label)
                if len(ybound) == 2:
                    videoid = self.data.single_frames[idy]['id']
                    videoid = int(videoid.split('-')[0])
                    frames = new_pseudo_label[videoid]
                    label = self.data.single_frames[idy]['label'][0]
                    for j in range(ybound[0], ybound[1] + 1):
                        if label not in frames[j]:
                            frames[j].append(label)

    def get_recall(self):
        assert self.pseudo_labels is not None

        idxs = self.data.train_vid
        gt = self.data.gt_labels
        labels = self.data.classlist

        res = []

        for i in range(len(labels)):
            all = 0
            true = 0
            for idx in idxs:
                frames = gt[idx]
                for j in range(len(frames)):
                    if i not in frames[j]:
                        continue
                    all += 1
                    if i in self.pseudo_labels[idx][j]:
                        true += 1
            res.append([true, all, true / all])
        return res

    def get_precision(self):
        assert self.pseudo_labels is not None

        idxs = self.data.train_vid
        gt = self.data.gt_labels
        labels = self.data.classlist

        res = []

        for i in range(len(labels)):
            all = 0
            true = 0
            for idx in idxs:
                frames = self.pseudo_labels[idx]
                for j in range(len(frames)):
                    if i not in frames[j]:
                        continue
                    all += 1
                    if i in gt[idx][j]:
                        true += 1
            res.append([true, all, true / all])
        return res

    def generate_svg(self, idx, idy):
        self.get_knn_pairs()
        # idx = int(key.split('-')[0])
        # idy = int(key.split('-')[1])
        if idx > idy:
            idx, idy = idy, idx
        key = str(idx) + "-" + str(idy)

        single_frame_labels = self.data.single_frame_labels
        whole_to_indvdual = self.data.whole_to_indvdual

        single_frames = self.data.single_frames
        single_framex = single_frames[idx]
        videoidx = int(single_framex['id'].split('-')[0])
        frameidx = int(single_framex['id'].split('-')[1])
        label = single_framex['label'][0]
        single_framex['lb'] = left_bound(single_frame_labels[videoidx], frameidx, label)
        single_framex['rb'] = right_bound(single_frame_labels[videoidx], frameidx, label)

        single_framey = single_frames[idy]
        videoidy = int(single_framey['id'].split('-')[0])
        frameidy = int(single_framey['id'].split('-')[1])
        label = single_framey['label'][0]
        single_framey['lb'] = left_bound(single_frame_labels[videoidy], frameidy, label)
        single_framey['rb'] = right_bound(single_frame_labels[videoidy], frameidy, label)

        pairs = self.knn_pairs_dict[key]
        pairs = [[whole_to_indvdual[p[0]], whole_to_indvdual[p[1]]] for p in pairs]
        self.draw_match(single_framex, single_framey, pairs, key)

    def draw_match(self, single_framex, single_framey, pairs, filename):
        videoidx = int(single_framex['id'].split('-')[0])
        videoidy = int(single_framey['id'].split('-')[0])

        buffer_dir = os.path.join(self.step_data_root, "image_buffer")
        check_dir(buffer_dir)
        filename = os.path.join(buffer_dir, filename)
        image_count = 0

        picture_width = 30
        pciture_height = 40
        gap = 10

        if videoidx == videoidy and single_framex['rb'] >= single_framey['lb']:
            width = (picture_width + gap) * (single_framey['rb'] - single_framex['lb']) + picture_width
            height = 1000
            d = draw.Drawing(width, height, displayInline=False)
            start = single_framex['lb']
            for i in range(single_framex['lb'] - start, single_framex['rb'] - start + 1):
                x = (picture_width + gap) * i
                path = self.data.get_single_frame(videoidx, i + start)
                d.append(draw.Image(x, height - pciture_height, picture_width, pciture_height, path=path))
                image_count += 1

            for i in range(single_framey['lb'] - start, single_framey['rb'] - start + 1):
                x = (picture_width + gap) * i
                path = self.data.get_single_frame(videoidy, i + start)
                d.append(draw.Image(x, 0, picture_width, pciture_height, path=path))
                image_count += 1

            for p in pairs:
                startx = int(p[0].split('-')[1]) - start
                end = int(p[1].split('-')[1]) - start
                d.append(draw.Line(startx * (picture_width + gap) + picture_width / 2, height - pciture_height, end * (picture_width + gap) + picture_width / 2, pciture_height, stroke_width=1, stroke='lime',
                            fill='black', fill_opacity=0.2))
            lgt = single_framex['lgt'][0] - start
            rgt = single_framex['rgt'][0] - start
            d.append(draw.Line(lgt * (picture_width + gap), 1000, lgt * (picture_width + gap), 500, stroke_width=1, stroke='red',
                            fill='black', fill_opacity=0.2))
            d.append(draw.Line(rgt * (picture_width + gap) + picture_width, 1000, rgt * (picture_width + gap) + picture_width, 500, stroke_width=1, stroke='red',
                            fill='black', fill_opacity=0.2))
            lgt = single_framey['lgt'][0] - start
            rgt = single_framey['rgt'][0] - start
            d.append(draw.Line(lgt * (picture_width + gap), 500, lgt * (picture_width + gap), 0, stroke_width=1, stroke='red',
                            fill='black', fill_opacity=0.2))
            d.append(draw.Line(rgt * (picture_width + gap) + picture_width, 500, rgt * (picture_width + gap) + picture_width, 0, stroke_width=1, stroke='red',
                            fill='black', fill_opacity=0.2))
            d.setPixelScale(2)
        else:
            widthx = single_framex['rb'] - single_framex['lb'] + 1
            widthy = single_framey['rb'] - single_framey['lb'] + 1
            width = (picture_width + gap) * max(widthx, widthy) - gap
            height = 1000
            d = draw.Drawing(width, height, displayInline=False)
            startx = single_framex['lb']
            starty = single_framey['lb']
            for i in range(single_framex['lb'] - startx, single_framex['rb'] - startx + 1):
                x = (picture_width + gap) * i
                path = self.data.get_single_frame(videoidx, i + startx)
                d.append(draw.Image(x, height - pciture_height, picture_width, pciture_height, path=path))
                image_count += 1

            for i in range(single_framey['lb'] - starty, single_framey['rb'] - starty + 1):
                x = (picture_width + gap) * i
                path = self.data.get_single_frame(videoidy, i + starty)
                d.append(draw.Image(x, 0, picture_width, pciture_height, path=path))
                image_count += 1

            for p in pairs:
                start = int(p[0].split('-')[1]) - startx
                end = int(p[1].split('-')[1]) - starty
                d.append(draw.Line(start * (picture_width + gap) + picture_width / 2, height - pciture_height, end * (picture_width + gap) + picture_width / 2, pciture_height, stroke_width=1, stroke='lime',
                            fill='black', fill_opacity=0.2))
            lgt = single_framex['lgt'][0] - startx
            rgt = single_framex['rgt'][0] - startx
            d.append(draw.Line(lgt * (picture_width + gap), 1000, lgt * (picture_width + gap), 500, stroke_width=1, stroke='red',
                            fill='black', fill_opacity=0.2))
            d.append(draw.Line(rgt * (picture_width + gap) + picture_width, 1000, rgt * (picture_width + gap) + picture_width, 500, stroke_width=1, stroke='red',
                            fill='black', fill_opacity=0.2))
            lgt = single_framey['lgt'][0] - starty
            rgt = single_framey['rgt'][0] - starty
            d.append(draw.Line(lgt * (picture_width + gap), 500, lgt * (picture_width + gap), 0, stroke_width=1, stroke='red',
                            fill='black', fill_opacity=0.2))
            d.append(draw.Line(rgt * (picture_width + gap) + picture_width, 500, rgt * (picture_width + gap) + picture_width, 0, stroke_width=1, stroke='red',
                            fill='black', fill_opacity=0.2))
            d.setPixelScale(2)
        print("filename", filename)
        # d.saveSvg(filename + '.svg')
        d.savePng(filename + '.png')
