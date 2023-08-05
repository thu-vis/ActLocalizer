import math
import os
from re import L
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

from IPython import embed
from copy import deepcopy

from application.views.model_utils.action_affinity_matrix import ActionAffinityMatrix
from ..utils.helper_utils import json_load_data, json_save_data, pickle_save_data, pickle_load_data
from .alignment import Alignment
from .frame_selector import FrameSelector
from .hierarchical_cluster import KMedoidsCluster, get_hierarchy_by_cluster, get_hierarchy_index, relayout_in_target_node, update_affinity_matrix_by_tree_and_lines
from .reorder import get_center_dists, get_order_by_dist_matrix_ctr, get_order_by_dist_matrix_ctr
from .y_layout import solve_y_layout_ctr

class ClusterHelper(object):
    use_align_cache = True

    def __init__(self, prop_model):
        # >>> cluster helper state and basic info
        # cluster helper state
        self.is_run = False
        # cluster helper info
        self.prop_model = prop_model
        self.dataname = self.prop_model.dataname
        self.step_data_root = self.prop_model.step_data_root
        self.previous_data_root = self.prop_model.previous_data_root
        # basic data
        self.affinity_matrix = None
        self.features = self.prop_model.data.get_all_features()
        self.action_classes = {}
        self.action_index_dict = {}
        self.track_and_field_idx = [9, 10, 11, 12, 13, 14, 17]
        # arguments
        self.alpha, self.beta = 0.9, 1
        # buffer path
        self.root_path = os.path.join(
            self.step_data_root, "cluster")
        self.buffer_path = os.path.join(
            self.step_data_root, "cluster", "buffer")
        self.similar_path = os.path.join(
            self.step_data_root, "cluster", "similarity")
        self.frame_path = os.path.join(
            self.step_data_root, "cluster", "frames")
        self.cache_path = os.path.join(
            self.step_data_root, "cluster", "cache")
        self.pre_cache_path = None if self.previous_data_root == None \
            else os.path.join(self.previous_data_root, "cluster", "cache")
        paths = [self.root_path, self.buffer_path, \
            self.similar_path, self.frame_path, self.cache_path]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # >>> key values and current cluster
        # meta info and cluster results
        self.meta_info = []
        self.clusters = {}
        self.key_frames = {}
        # current hierarchy
        self.cur_index = -1
        self.cur_hierarchy = {} 
        self.all_orders = {}
        self.all_centers = {}
        # for local update
        self.all_lines = {}           # (saved) node name -> line
        self.all_adj_dists = {}       # (saved) node name -> adj dists
        self.all_rep_info = {}        # (saved) node name -> rep info
        self.node_indexes = {}        # node name -> node 
        self.parent_indexes = {}      # node name -> parent name
        # for remove list
        self.remove_list = set()

    # 0. init cluster helper 
    def run(self):
        self.init()
        # self.gen_hierarchy()

    def load_prerun_result(self):
        self.init()
        self.load_hierarchy_meta()
            
    def init(self):
        self.get_pseudo_actions()
        self.load_cluster()
        self.load_key_frames()
        self.is_run = True

    def update(self):
        # for Debug
        # self.tmp = {}
        # for cdx in range(19):
        #     for action in self.action_classes[cdx]:
        #         self.tmp[action['id']] = [action['bound'][0], action['bound'][1]]
        # self.tmp['ok'] = True
        self.get_pseudo_actions()
        # self.unequal = [0, 0]
        # self.unequalgap = [[], []]
        # for cdx in range(19):
        #     for action in self.action_classes[cdx]:
        #         action['old_bound'] = self.tmp[action['id']]
        #         if action['old_bound'][0] != action['bound'][0]:
        #             self.unequal[0] += 1
        #             self.unequalgap[0].append(action['old_bound'][0] - action['bound'][0])
        #         if action['old_bound'][1] != action['bound'][1]:
        #             self.unequal[1] += 1
        #             self.unequalgap[1].append(action['bound'][1] - action['old_bound'][1])
        # embed(); exit()
        self.load_key_frames(update=True)
        self.gen_class_hierarchy(self.cur_index, update=True)
        save_data = {
            "clusters": self.clusters, 
            "cur_index": self.cur_index,
            "hierarchy": self.cur_hierarchy,
            "save_info": self.save_info,
            "key_frames": self.key_frames,
            "remove_list": self.remove_list,
            "meta_info": self.meta_info
        }
        return save_data

    def remove(self, aid):
        self.remove_list.add(aid)
    
    def load_from_predecessor(self, save_data):
        # save pre info 
        pickle_save_data(os.path.join(self.buffer_path, "clusters.pkl"), save_data['clusters'])
        json_save_data(os.path.join(self.buffer_path, str(save_data['cur_index']) + ".json"), save_data['hierarchy'])
        pickle_save_data(os.path.join(self.buffer_path, str(save_data['cur_index']) + "_info.pkl"), save_data['save_info'])
        pickle_save_data(os.path.join(self.root_path, "key_frames.pkl"), save_data['key_frames'])
        pickle_save_data(os.path.join(self.root_path, "remove_list.pkl"), save_data['remove_list'])

        # init and update cur_class entropy
        self.init()
        # for meta in save_data['meta_info']:
        #     if meta['name'] == save_data['cur_index']:
        #         meta['entropy'] = self.get_class_entropy(save_data['cur_index'])
        json_save_data(os.path.join(self.buffer_path, "meta.json"), save_data['meta_info'])
        self.meta_info = save_data['meta_info']
        self.load_hierarchy_i(save_data['cur_index'])

    def get_pseudo_actions(self):
        self.action_classes = {}
        self.action_index_dict = {}
        for action in self.prop_model.get_all_pseudo_actions():
            action['bound'] = self.get_action_bound(action)
            for cdx in action['label']:
                if cdx not in self.action_classes:
                    self.action_classes[cdx] = []
                    self.action_index_dict[cdx] = {}
                    self.action_index_dict[cdx]['count'] = 0
                self.action_classes[cdx].append(action)
                self.action_index_dict[cdx][action['id']] = self.action_index_dict[cdx]['count']
                self.action_index_dict[cdx]['count'] += 1

    def call_test(self):
        self.load_prerun_result()
        
        # >>> generate cluster
        # self.gen_cluster()

        # >>> generate hierarchy
        # self.gen_hierarchy()
        self.gen_class_hierarchy(10)

        # >>> calculate metrics
        # result = []
        # for cdx in self.track_and_field_idx:
        #     self.gen_class_hierarchy(cdx)
        #     result.append((self.get_class_avg_pred_scores(cdx), cdx))
        # result.sort()
        # for r, i in result:
        #     print("[{}] {}.".format(i, r))
        
        # >>> load and update hierarchy
        # self.load_hierarchy_i(10)
        # constraint = {}
        # constraint['node_name'] = 'M10-8'
        # constraint['child_index1'] = 0
        # constraint['child_index2'] = 1
        # self.update_cur_hierarchy_by_constraint(constraint)
        # # embed()
        
        # >>> anchor action alignments
        # e.g. 2, 15   2, 20   2, 25  2，47  2，49 3, 6/7/8
        # class_id, action_id = 10, 32
        # acs = self.action_classes[class_id]
        # aid = acs[action_id]['id']
        # print(self.get_alignment_of_anchor_action(class_id, aid, 5))
        return
    
    # 1. functions for cluster
    def gen_cluster(self):
        clusters = {}
        for cdx in self.action_classes.keys():
            kc = KMedoidsCluster(cdx, dist_matrix=self.get_precomputed_matrix(cdx))
            clusters[cdx] = kc.do_cluster()
        pickle_save_data(os.path.join(self.buffer_path, "clusters.pkl"), clusters)
        self.clusters = clusters

    def load_cluster(self):
        cluster_file = os.path.join(self.buffer_path, "clusters.pkl")
        if os.path.exists(cluster_file):
            self.clusters = pickle_load_data(cluster_file)
        else:
            self.gen_cluster()
        remove_file = os.path.join(self.root_path, "remove_list.pkl")
        if os.path.exists(remove_file):
            self.remove_list = set(pickle_load_data(remove_file))

    # 2. functions for hierarchy generation
    def load_key_frames(self, reload=False, update=False):
        key_frames_path = os.path.join(self.root_path, "key_frames.pkl")
        if not update and not reload and os.path.exists(key_frames_path):
            self.key_frames = pickle_load_data(key_frames_path)
            return
        key_frames = {}
        print("Cluster helper: generate key frames buffer, need around 150 seconds....")
        selector = FrameSelector(self.prop_model)
        for cdx in self.action_classes.keys():
            for action in self.action_classes[cdx]:
                video_id = int(action['id'].split('-')[0])
                frames = selector.select_frames(
                    video_id, action['left_bound'], action['right_bound'], 100)
                key_frames[action['id']]= list(map(lambda x: x + action['left_bound'], frames))
        self.key_frames = key_frames
        if not update:
            pickle_save_data(key_frames_path, key_frames)


    def gen_affinity_matrix(self, cdx):
        affinity_matrix = self.prop_model.get_symmetrical_affinity_matrix()
        action_frame_set = set()
        for action in self.action_classes[cdx]:
            bound = action['bound']
            for frame in range(bound[0], bound[1] + 1):
                action_frame_set.add(frame)
        action_affinity_matrix = ActionAffinityMatrix(cdx, action_frame_set, affinity_matrix)
        return action_affinity_matrix

    def gen_hierarchy(self):
        assert self.is_run
        self.load_hierarchy_meta()
        # self.load_key_frames(reload=True)
        for cdx in self.action_classes.keys():
            self.gen_class_hierarchy(cdx)

    def gen_hierarchy_meta(self):
        assert self.is_run
        meta_file = os.path.join(self.buffer_path, "meta.json")
        self.meta_info = []
        for cdx in self.action_classes.keys():
            meta = {}
            meta['name'] = cdx
            meta['labeled_num'] = len(self.action_classes[cdx])
            meta['entropy'] = self.get_class_entropy(cdx)
            self.meta_info.append(meta)
        json_save_data(meta_file, self.meta_info)

    def gen_class_hierarchy(self, cdx, update=False):
        assert self.is_run
        print("[{}] Cluster Helper: Generate hierarchy for class {}.".format(cdx, cdx))
        time_start = time.time()
        self.affinity_matrix = self.gen_affinity_matrix(cdx)
        tree, save_info = get_hierarchy_by_cluster(cdx, self.clusters[cdx], self.action_classes[cdx], 
                self, self.get_precomputed_matrix(cdx), use_pre_order=update)
        if not update:
            json_save_data(os.path.join(self.buffer_path, str(cdx) + ".json"), tree)
            pickle_save_data(os.path.join(self.buffer_path, str(cdx) + "_info.pkl"), save_info)
        else:
            self.cur_hierarchy = tree
            self.save_info = save_info
        time_end = time.time()
        print("[{}] Cluster Helper: Finish hierarchy using time {} s.".format(
            cdx, time_end-time_start))
        return time_end-time_start

    # 3. functions for hierarchy incremental update
    def load_hierarchy_meta(self):
        meta_file = os.path.join(self.buffer_path, "meta.json")
        if not os.path.exists(meta_file):
            self.gen_hierarchy_meta()
        else:
            self.meta_info = json_load_data(meta_file)
        return self.meta_info

    def load_hierarchy_i(self, i):
        if self.cur_index == i:
            return self.cur_hierarchy

        hierarchy_file = os.path.join(self.buffer_path, str(i) + ".json")
        if not os.path.exists(hierarchy_file):
            self.gen_class_hierarchy(i)
        self.cur_index = i
        self.cur_hierarchy = json_load_data(os.path.join(self.buffer_path, str(i) + ".json"))
        hierarchy_info = pickle_load_data(os.path.join(self.buffer_path, str(i) + "_info.pkl"))  
        self.all_orders = hierarchy_info['all_orders']
        self.all_centers = hierarchy_info['all_centers']   
        # self.all_lines = hierarchy_info['all_lines']
        # self.all_adj_dists = hierarchy_info['all_adj_dists']
        # self.node_indexes, self.parent_indexes = get_hierarchy_index(self.cur_hierarchy)
        return self.cur_hierarchy

    def update_cur_hierarchy_by_constraint(self, constraint):
        """
        Update current hierarchy by constraint when change.
        Params:
            constraint - Dict
                node_name - parent node name
                child_index1 - index 1
                child_index2 - index 2
        Return:
            cur_hierarchy - current hierarchy
        """
        # Analyse constraint data and check data accuracy
        assert self.cur_index > -1
        node_name = constraint['node_name']
        child_index1, child_index2 = constraint['child_index1'], constraint['child_index2']
        assert node_name in self.node_indexes
        node = self.node_indexes[node_name]
        ln = len(node['children'])
        assert child_index1 < ln and child_index2 < ln and child_index1 != child_index2

        # Generate and update affinity_matrix by lines
        self.affinity_matrix = self.gen_affinity_matrix(self.cur_index)
        update_affinity_matrix_by_tree_and_lines(self.affinity_matrix, self.cur_hierarchy, self.all_lines)

        # Realign and layout for current node
        relayout_in_target_node(node, [child_index1, child_index2], self)

        # Update parent node until root node
        print(node['name'])
        while node['name'] in self.parent_indexes:
            parent_name, index = self.parent_indexes[node['name']]
            node = self.node_indexes[parent_name]
            print(node['name'])
            relayout_in_target_node(node, [index], self)
        
        # Save constraint
        self.constraints.append(constraint)
        return self.cur_hierarchy

    # 4. functions for anchor/center mode
    def decode_pair(self, idx, pair):
        idx1, idx2 = pair.split('-')
        idx1, idx2 = int(idx1), int(idx2)
        if idx1 == idx:
            return idx2
        return idx1

    def get_action(self, cdx, aid):
        assert self.is_run
        assert aid in self.action_index_dict[cdx]
        index = self.action_index_dict[cdx][aid]
        return self.action_classes[cdx][index]

    def get_alignment_of_anchor_action(self, cdx, aid, k):
        """
        Get k-nn as alignments
        """
        assert self.is_run
        assert aid in self.action_index_dict[cdx]
        
        # 0. judge buffer or get buffer
        cache_path = os.path.join(self.cache_path, "{}_.pkl".format(aid))
        # read align buffer
        if ClusterHelper.use_align_cache and os.path.exists(cache_path):
            print("Cluster Helper: Use buffer for anchor alignment of action {}.".format(aid))
            ret = pickle_load_data(cache_path)["ret"]
            return ret

        self.affinity_matrix = self.gen_affinity_matrix(cdx)

        # 1. get knn actions
        # if k == -1:
        #     k = 3   # TODO: get default k
        # index = self.action_index_dict[cdx][aid]
        # dist = self.get_precomputed_matrix(cdx)
        # knn_indexs = np.argsort(dist[index])[0:k+1]
        # actions = []#[self.action_classes[cdx][i]]
        # for i in knn_indexs:
        #     actions.append(self.action_classes[cdx][i])
        actions = []
        index = self.action_index_dict[cdx][aid]
        actions.append(self.action_classes[cdx][index])
        single_idx = self.action_classes[cdx][index]['single_idx']
        knn_pair = self.prop_model.get_knn_pair(single_idx)
        k = len(knn_pair)
        if k > 7:
            k = 7
            knn_pair = knn_pair[:7]
        for pair in knn_pair:
            _single_idx = self.decode_pair(single_idx, pair)
            action_id = self.prop_model.get_single_frame_by_single_idx(_single_idx)['id']
            action_index = self.action_index_dict[cdx][action_id]
            actions.append(self.action_classes[cdx][action_index])
        

        # 2. do alignment
        ac = Alignment(0, actions, self)
        lines = list(map(lambda x: x.copy(), ac.alignments))
        ln = len(lines)
        ac.cluster_to(1)

        idx_dict, idx = {}, 0
        for action in actions:
            idx_dict[action['idx']] = idx
            idx += 1
        aligns = np.zeros((ln, ac.alignments[0].width), dtype='int32')
        for i in range(ln):
            idx = ac.alignments[0].ids[i]
            pos = idx_dict[idx]
            aligns[pos] = ac.alignments[0].frames[i]
        _aligns = []
        for align in aligns:
            _aligns.append(list(map(lambda x: -1 if x == -1 else 1, align)))
        
        # 3. calculate distances and do reorder
        dists = np.array(get_center_dists(lines, aligns, self.affinity_matrix))
        vis_matrix = np.zeros(dists.shape)
        for i in range(k):
            vec = np.nonzero(np.array(_aligns[i+1]) + 1)[0]
            lf, rt = vec.min(), vec.max() + 1
            vis_matrix[i][lf: rt] = 1

        history_dists = None
        history_order = None
        history_vis = None
        if self.pre_cache_path:
            history_path = os.path.join(self.pre_cache_path, "{}_.pkl".format(aid))
            if os.path.exists(history_path):
                pre_save = pickle_load_data(history_path)
                cur_actions = list(map(lambda x: x['id'], actions))[1:]
                his_actions = pre_save["ret"]["action_list"]
                if len(cur_actions) == len(his_actions):
                    use_history = True
                    for i in range(len(cur_actions)):
                        if cur_actions[i] != his_actions[i]:
                            use_history = False
                            break
                    if use_history:
                        history_dists = pre_save["dists"]
                        history_order = pre_save["order"]
                        history_vis = pre_save["vis"]

        total_zero = False
        if np.sum(dists) == 0:
            total_zero = True
            _dists = dists.copy()
            l = dists.shape[1]
            dists = np.vstack((dists, np.ones((1, l))))
            order = []
            if history_order:
                line_order = list(deepcopy(history_order[0]))
                line_order.append(len(history_order[0]))
                order = [tuple(line_order) for _ in range(l)] 
                _order = [tuple(history_order[0]) for _ in range(l)]
            else:
                order = [tuple(range(dists.shape[0])) for _ in range(l)]
                _order = [tuple(range(dists.shape[0] - 1)) for _ in range(l)]
        else:
            # Pre process and Reorder: clear zero lines
            order = []
            if history_order:
                his_order_matrix = np.array(history_order)
                pre_list, pre_index = [], 0
                while True:
                    if np.sum(his_order_matrix[:, pre_index] - his_order_matrix[0][pre_index]) == 0\
                        and np.sum(dists[his_order_matrix[0][pre_index]]) == 0:
                        pre_list.append(his_order_matrix[0][pre_index])
                        pre_index += 1
                    else:
                        break
                tmp_order = get_order_by_dist_matrix_ctr(dists, vis_matrix)
                for tmp_line in tmp_order:
                    line = pre_list[:]
                    for num in tmp_line:
                        if num not in pre_list:
                            line.append(num)
                    order.append(line)                
            else:
                order = get_order_by_dist_matrix_ctr(dists, vis_matrix)

            # Post process: adjust order
            if history_order:
                ## 1. find key line
                target = -1
                bound = []
                for i in range(dists.shape[0]):
                    his_dist = history_dists[i]
                    his_vis = history_vis[i]
                    cur_dist = dists[i]
                    if his_dist.sum() != 0 and cur_dist.sum() == 0:
                        target = i
                        nonzero = his_vis.nonzero()[0]
                        bound = [nonzero.min(), nonzero.max() + 1]
                        break
                ## 2. judge crosses
                if target != -1:
                    indexes = []
                    for his_order in history_order[bound[0]: bound[1]]:
                        indexes.append(list(his_order).index(target))
                    pos = np.argmax(np.bincount(np.array(indexes)))  
                    while pos > -1:
                        # search suitable pos
                        do_adjust = True
                        tmp_orders = []
                        print("xxx    ", pos)
                        for j in range(dists.shape[1]):
                            cur_order = list(order[j])
                            cur_pos = cur_order.index(target)
                            if cur_pos >= pos:
                                for ptr in range(pos, cur_pos):
                                    cur_order[ptr + 1] = order[j][ptr]
                                cur_order[pos] = target
                            else:
                                final_pos = pos
                                for ptr in range(cur_pos, pos):
                                    if vis_matrix[order[j][ptr + 1]][j] and dists[order[j][ptr + 1]][j] > 0 or \
                                        j > 0 and vis_matrix[order[j][ptr + 1]][j - 1] and dists[order[j][ptr + 1]][j - 1] > 0 or \
                                        j < dists.shape[1] - 1 and vis_matrix[order[j][ptr + 1]][j + 1] and dists[order[j][ptr + 1]][j + 1] > 0:
                                        if vis_matrix[target][j]:
                                            do_adjust = False
                                            final_pos = ptr
                                            break
                                    if vis_matrix[order[j][ptr + 1]][j] and dists[order[j][ptr + 1]][j] > 0:
                                        final_pos = ptr
                                        break
                                    cur_order[ptr] = order[j][ptr + 1]
                                cur_order[final_pos] = target
                            tmp_orders.append([j, cur_order])
                        
                        if do_adjust:
                            for j, cur_order in tmp_orders:
                                order[j] = cur_order
                            break
                        else:
                            pos -= 1

        # 4. do y_layout
        y_layout = solve_y_layout_ctr(dists, order)
        y_layout = np.array(y_layout)
        # y_layout = np.concatenate((np.zeros((1, y_layout.shape[1])), y_layout * 0.8 + 0.2), axis = 0)
        y_layout = np.concatenate((np.zeros((1, y_layout.shape[1])), y_layout), axis = 0)
        y_layout = y_layout[: k+1]

        # 5. return value
        ret = {}
        ret['class'] = cdx
        ret['action_id'] = aid
        ret['action_list'] = list(map(lambda x: x['id'], actions))[1:]
        ret['y_layout'] = y_layout.tolist()
        ret['aligns'] = _aligns
        ret['single_frames'] = []
        for i in range(ln):
            align = _aligns[i]
            single_index = actions[i]['single_frame'] - actions[i]['left_bound']
            cur_index = -1
            for pos in range(len(align)):
                if align[pos] > -1:
                    cur_index += 1
                    if cur_index == single_index:
                        ret['single_frames'].append(pos)
        ret['actions'] = []
        for action in actions:
            node = {}
            node['name'] = action['id']
            node['bound'] = [action['left_bound'], action['right_bound']]
            node['video_id'] = action['video_id']
            node['single_frame'] = action['single_frame']
            node['pred_scores'] = action['pred_scores']
            node['key_frames'] = self.key_frames[node['name']]
            ret['actions'].append(node)
        
        save = {}
        save["ret"] = ret
        save["dists"] = dists if not total_zero else _dists
        save["order"] = order if not total_zero else _order
        save["vis"] = vis_matrix
        pickle_save_data(cache_path, save)
        return ret

    # 5. functions for assistment
    def get_action_bound(self, action):
        b = int(action['id'].split('-')[1])
        return [action['idx'] - b + action['left_bound'], action['idx'] - b + action['right_bound']]

    def get_precomputed_matrix(self, cdx):
        dist_matrix = 1 - np.load(os.path.join(
            self.similar_path, str(cdx) + ".npy"))
        row, col = np.diag_indices_from(dist_matrix)
        dist_matrix[row, col] = 0
        return dist_matrix

    def save_action_frames(self, cdx):
        actions = self.action_classes[cdx]
        path = os.path.join(self.frame_path, 'class {}'.format(cdx))
        if not os.path.exists(path):
                os.makedirs(path)

        for action in actions:
            ls = [action['left_bound'], action['right_bound']]
            if ls[1] + 1 - ls[0] <= 5:
                fs = list(range(action['left_bound'], action['right_bound'] + 1))
            else:
                delta = math.ceil((ls[1] + 1 - ls[0]) / 5)
                fs = list(range(action['left_bound'], action['right_bound'] + 1, delta))

            action_path = os.path.join(path, action['id'])
            if not os.path.exists(action_path):
                os.makedirs(action_path)
            video_id = int(action['id'].split('-')[0])
            for f in fs:
                frame = self.prop_model.get_single_frame(video_id, f)
                cv2.imwrite(os.path.join(action_path, '{}.jpg'.format(f)), frame)

    def get_class_entropy(self, cdx):
        actions = self.action_classes[cdx]
        class_entro = 0
        total_num = 0
        for action in actions:
            b = int(action['id'].split('-')[1])
            bound = [action['idx'] - b + action['left_bound'], action['idx'] - b + action['right_bound']]
            action_entro, action_num = 0, 0 
            for frame in range(bound[0], bound[1] + 1):
                entro = entropy(self.prop_model.propagation_prediction[frame])
                action_entro += entro
                action_num += 1
            # action_entro /= action_num
            total_num += action_num
            class_entro += action_entro
        return class_entro / total_num

    def get_class_avg_pred_frames(self, cdx):
        actions = self.action_classes[cdx]
        class_pred = 0
        for action in actions:
            width = action['right_bound'] - action['left_bound'] + 1
            pseudo_width = action["right_pseudo"] - action["left_pseudo"] + 1
            class_pred += pseudo_width
        return class_pred / len(actions) 

def entropy(data):
    ret = 0
    for value in data:
        if value > 0:
            ret -= value * math.log2(value)
    return ret


