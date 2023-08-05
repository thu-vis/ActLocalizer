from __future__ import print_function
import enum
from re import I
from cv2 import add
import argparse
import os
import numpy as np
from ..model import SFNET
from ..dataset import Dataset
from ..dataset.config import config
from datetime import datetime
from copy import deepcopy
from scipy import io as sio
from scipy.special import softmax
from ..model import CenterLoss
from ..utils import get_logger, save_checkpoint, resume_checkpoint
from ..utils import check_dir, common_data
from ..kNN_utils import get_features, feature_norm
from ..kNN_utils import left_bound, right_bound, acquire_boundaries
from ..kNN_utils import encoding_pair, decoding_pair, construct_csr_matrix_by_edge
from ..kNN_utils import graph_propagation
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import networkx as nx
from time import time
import pickle
import json
from ..dataset.video_dataset import left_active, right_active
import SF_Net.options
try:
    import torch
    import torch.optim as optim
    from tensorboard_logger import Logger as TB_Writer
    from torch.autograd import Variable
except:
    None

def adaptive_knn(trainX, trainY, testx, kmax, A, simi=True):
    # trainX: (N, feature_size) array
    # trainY: (N, num_classes) array
    # testx: (1, feature_size) array
    # kmax: int
    # A: 实际使用时的参数
    # simi: feature之间计算相似度True，计算距离为False
    N = trainX.shape[0]
    if simi:
        dis = np.dot(trainX, testx.T)
        sorted_idxs = dis.argsort()
        sorted_idxs = sorted_idxs[::-1]
        # 相似度大排序
    else:
        dis = trainX - testx
        dis = dis * dis
        dis = np.sum(dis, axis=1)
        sorted_idxs = dis.argsort()
        # 距离小排序
    Y = trainY[sorted_idxs]
    # sort the labels

    k = 2
    eta = np.average(Y[:k,:], axis=0)
    while k <= kmax:
        if eta.max() > A * np.power(k, -0.5):
            return k
        else:
            k += 1
            eta = np.average(Y[:k,:], axis=0)
    return kmax

def save_linechart(dataset, video_idx, tcam):
    videoname = dataset.videoname[video_idx].decode()
    frame_label = dataset.get_init_frame_label(video_idx)
    gt_label = dataset.get_gt_frame_label(video_idx)
    video_dir = os.path.join("./test/expand/result", str(video_idx) + "-" + videoname)
    check_dir(video_dir)
    anchor_frames = [
        i for i in range(len(frame_label)) if len(frame_label[i]) > 0
    ]
    for i in range(len(anchor_frames)):
        idx = anchor_frames[i]
        anchor_label = frame_label[idx]
        gt = [5 if len(set(anchor_label).intersection(i)) > 0 else 0 for i in gt_label]
        pred = tcam[:,anchor_label].reshape(-1)
        plt.plot(pred)
        plt.plot(gt, c="b")
        plt.axvline(idx, c="r")
        plt.savefig(os.path.join(video_dir, str(idx) + ".jpg"))
        plt.close()
    # import IPython; IPython.embed(); exit()

def pair_detailed_min_distance(feature_pair_1, feature_pair_2, pred_pair_1, pred_pair_2, kmax, A, constraints):
    # t0 = time()
    simi = np.dot(feature_pair_1, feature_pair_2.T)
    nodes = []
    same = []
    if 'same' in constraints.keys():
        same = constraints['same']
    set_k = None
    if 'k' in constraints.keys():
        set_k = constraints['k']

    must_not_match = [[], []]
    if 'must_not_match' in constraints.keys():
        must_not_match = constraints['must_not_match']

    must_match = []
    ends1 = []
    ends2 = []

    # constraints['must_match'] = ['12-5']
    if 'must_match' in constraints.keys():
        must_match = constraints['must_match']
        for item in must_match:
            ends1.append(item[0])
            ends2.append(item[1])
            nodes.append(str(item[0]) + '-' + str(item[1]))
        ends1.append(len(feature_pair_1))
        ends2.append(len(feature_pair_2))
        start1 = 0
        start2 = 0
        for k in range(len(ends1)):
            end1 = ends1[k]
            end2 = ends2[k]
            for i in range(start1, end1):
                if set_k is None:
                    topk = adaptive_knn(feature_pair_2[start2: end2], pred_pair_2[start2: end2], feature_pair_1[i], kmax, A)
                else:
                    topk = set_k
                d = simi[i, :]
                sorted_idxs = d.argsort()[::-1]
                count = 0
                for j in sorted_idxs:
                    if encoding_pair(i, j) not in same and start2 <= j < end2:
                        nodes.append(encoding_pair(i, j))
                        count += 1
                    if count == topk:
                        break

            for j in range(start2, end2):
                if set_k is None:
                    topk = adaptive_knn(feature_pair_1[start1: end1], pred_pair_1[start1: end1], feature_pair_2[j], kmax, A)
                else:
                    topk = set_k
                d = simi[:, j]
                sorted_idxs = d.argsort()[::-1]
                count = 0
                for i in sorted_idxs:
                    if encoding_pair(i, j) not in same and start1 <= i < end1:
                        nodes.append(encoding_pair(i, j))
                        count += 1
                    if count == topk:
                        break
            start1 = end1
            start2 = end2
    else:
        for i in range(len(feature_pair_1)):
            if set_k is None:
                topk = adaptive_knn(feature_pair_2, pred_pair_2, feature_pair_1[i], kmax, A)
            else:
                topk = set_k
            d = simi[i, :]
            sorted_idxs = d.argsort()[::-1]
            count = 0
            for j in sorted_idxs:
                if encoding_pair(i, j) not in same:
                    nodes.append(encoding_pair(i, j))
                    count += 1
                if count == topk:
                    break

        for j in range(len(feature_pair_2)):
            if set_k is None:
                topk = adaptive_knn(feature_pair_1, pred_pair_1, feature_pair_2[j], kmax, A)
            else:
                topk = set_k
            d = simi[:, j]
            sorted_idxs = d.argsort()[::-1]
            count = 0
            for i in sorted_idxs:
                if encoding_pair(i, j) not in same:
                    nodes.append(encoding_pair(i, j))
                    count += 1
                if count == topk:
                    break
    nodes = list(set(nodes))
    window_size = 1000
    if 'window_size' in constraints.keys():
        window_size = int(constraints['window_size'])
    

    lb = [0, 100000]
    rb = [0, 100000]
    if 'boundary' in constraints.keys():
        if len(constraints['boundary'][0]) != 0:
            lb = constraints['boundary'][0]
        if len(constraints['boundary'][1]) != 0:
            rb = constraints['boundary'][1]

    nodes_cut = []
    for a in nodes:
        a_1, a_2 = decoding_pair(a)        
        if lb[0] <= a_1 <= lb[1] and rb[0] <= a_2 <= rb[1]:
            if a_1 not in must_not_match[0] and a_2 not in must_not_match[1]:
                nodes_cut.append(a)
    nodes = nodes_cut.copy()

    must_match_str = []
    for item in must_match:
        must_match_str.append(str(item[0]) + '-' + str(item[1]))
        nodes.append(str(item[0]) + '-' + str(item[1]))
    must_match = must_match_str
    nodes = list(set(nodes))

    t = nx.DiGraph()
    t.add_node("source", demand=-1)
    t.add_node("sink", demand=1)
    for a in nodes:
        a_1, a_2 = decoding_pair(a)
        w = simi[a_1, a_2]
        for b in nodes:
            if a == b:
                continue
            b_1, b_2 = decoding_pair(b)
            if 0 <= b_1 - a_1 <= window_size and 0 <= b_2 - a_2 <= window_size:
                t.add_edge(a, b, weight= w, capacity=1)
        t.add_edge("source", a, weight=0, capacity=1)
        t.add_edge(a, "sink", weight= w, capacity=1)
    
    if len(must_match) == 0:  
        res = nx.dag_longest_path(t)
        pairs = res[1:-1]
    elif len(must_match) >= 1:
        try:
            start = must_match[0]
            end = must_match[-1]

            old_weight = t.edges[start, 'sink']['weight']
            t.edges[start, 'sink']['weight'] = 100000
            res_start = nx.dag_longest_path(t)
            pairs_start = res_start[1:-2]
            t.edges[start, 'sink']['weight'] = old_weight

            for i in range(len(must_match) - 1):
                pre = must_match[i]
                next = must_match[i + 1]
                old_weight_pre = t.edges['source', pre]['weight']
                t.edges['source', pre]['weight'] = 100000
                old_weight_next = t.edges[next, 'sink']['weight']
                t.edges[next, 'sink']['weight'] = 100000
                res = nx.dag_longest_path(t)
                pairs = res[1:-2]
                pairs_start.extend(pairs)

                t.edges['source', pre]['weight'] = old_weight_pre
                t.edges[next, 'sink']['weight'] = old_weight_next

            old_weight = t.edges['source', end]['weight']
            t.edges['source', end]['weight'] = 100000
            res = nx.dag_longest_path(t)
            t.edges['source', end]['weight'] = old_weight
            pairs = res[1:-1]
            pairs_start.extend(pairs)
            pairs = pairs_start
        except:
            pairs = []

    cost = 0
    for pair in pairs:
        s = int(pair.split("-")[0])
        e = int(pair.split("-")[1])
        cost += simi[s, e]

    return pairs, cost

def pair_matching(all_features, all_preds, idvdual_to_whole, pair_1, pair_2, kmax, A, near, constraints={}):
    if 'boundary' in constraints.keys():
        boundary = constraints['boundary']
        assert len(boundary) == 2
        # two actions, if one action don't need to change boundary, use []
        left_boundary = boundary[0]
        right_boundary = boundary[1]
        if len(left_boundary) > 0:
            assert len(left_boundary) == 2
            constraints['boundary'][0][0] -= pair_1["lb"]
            constraints['boundary'][0][1] -= pair_1["lb"]
            # pair_1["lb"] = left_boundary[0]
            # pair_1["rb"] = left_boundary[1]
        if len(right_boundary) > 0:
            assert len(right_boundary) == 2
            constraints['boundary'][1][0] -= pair_2["lb"]
            constraints['boundary'][1][1] -= pair_2["lb"]
            # pair_2["lb"] = right_boundary[0]
            # pair_2["rb"] = right_boundary[1]
        
    if 'must_match' in constraints.keys():
        must_match = constraints['must_match']
        for item in must_match:
            item[0] -= pair_1["lb"]
            item[1] -= pair_2["lb"]

    if 'must_not_match' in constraints.keys():
        must_not_match = constraints['must_not_match']
        part0 = []
        part1 = []
        for item in must_not_match[0]:
            item -= pair_1["lb"]
            part0.append(item)

        for item in must_not_match[1]:
            item -= pair_2["lb"]
            part1.append(item)
        
        constraints['must_not_match'] = [part0, part1]

    feature_pair_1 = []
    pred_pair_1 = []
    if near:
        window_size = 3
        if pair_2['lb'] <= pair_1['rb']:
            start = pair_2['lb']
            end = pair_1['rb']
            constraints['same'] = []
            for i in range(start, end + 1):
                pair1_begin = i - pair_1['lb']
                pair2_begin = max(i - start - window_size, 0)
                pair2_end = min(i - start + window_size, end - pair_2['lb'])
                for k in range(pair2_begin, pair2_end + 1):
                    key = str(pair1_begin) + '-' + str(k)
                    constraints['same'].append(key)

    for f in range(pair_1["lb"], pair_1["rb"] + 1):
        frame_id = str(pair_1["v_id"]) + "-" + str(f)
        feature_pair_1.append(all_features[idvdual_to_whole[frame_id]])
        pred_pair_1.append(all_preds[idvdual_to_whole[frame_id]])
    feature_pair_1 = np.array(feature_pair_1)
    pred_pair_1 = np.array(pred_pair_1)
    feature_pair_2 = []
    pred_pair_2 = []
    for f in range(pair_2["lb"], pair_2["rb"] + 1):
        frame_id = str(pair_2["v_id"]) + "-" + str(f)
        feature_pair_2.append(all_features[idvdual_to_whole[frame_id]])
        pred_pair_2.append(all_preds[idvdual_to_whole[frame_id]])
    feature_pair_2 = np.array(feature_pair_2)
    pred_pair_2 = np.array(pred_pair_2)
    pairs, cost = pair_detailed_min_distance(feature_pair_1, feature_pair_2, pred_pair_1, pred_pair_2, kmax, A,
    constraints)
    pairs = [(int(i.split("-")[0]) + pair_1["lb"], int(i.split("-")[1]) + pair_2["lb"]) for i in pairs]
    pairs = [(str(pair_1["v_id"]) + "-" + str(p[0]), str(pair_2["v_id"]) + "-" + str(p[1])) for p in pairs]
    return pairs, cost

def graph_construction(dataset, all_features, idvdual_to_whole, labeled_frames, all_preds, A, buffer_path):
    # all_features, whole_to_invdual, idvdual_to_whole, labeled_frames = get_features(dataset)
    # all_features = feature_norm(all_features)
    # print("total features:", len(all_features))
    labeled_features = [i["idx"] for i in labeled_frames]
    labeled_preds = all_preds[np.array(labeled_features)]
    labeled_features = all_features[np.array(labeled_features)]
    dir = buffer_path + '/' + dataset.dataset_name + '/' + str(dataset.active_ratio)
    check_dir(dir)
    path = os.path.join(dir, 'simi_matrix.npy')
    if os.path.exists(path):
        simi_matrix = np.load(path)
        path = os.path.join(dir, 'pairs.json')
        with open(path, 'r') as f:
            pairs_dict = json.load(f)
    else: 
        simi_matrix, pairs_dict = store_similarity(dataset, all_features, all_preds, idvdual_to_whole, labeled_frames, A)
    # aggregate the features of labeled actions
    for idx, labeled_frame in enumerate(labeled_frames):
        labeled_id = labeled_frame["id"]
        labeled_video_id, labeled_frame_id = labeled_id.split("-")
        labeled_video_id = int(labeled_video_id)
        labeled_frame_id = int(labeled_frame_id)
        labeled_lb, labeled_rb, labeled_is_active, _ = acquire_boundaries(dataset, \
            labeled_video_id, labeled_frame_id)
        if labeled_is_active:
            ac_features = []
            for f in range(labeled_lb, labeled_rb + 1):
                frame_id = str(labeled_video_id) + "-" + str(f)
                ac_features.append(all_features[idvdual_to_whole[frame_id]])
            feature = np.array(ac_features).mean(axis=0)
            labeled_features[idx] = feature
        
    # construction
    kmax = 7
    labeled_distance = pairwise_distances(labeled_features)
    edges = []
    for count_idx, labeled_frame in tqdm(enumerate(labeled_frames)):
        # print("count_idx: ", count_idx)
        test_feature = labeled_features[count_idx]
        topk = adaptive_knn(labeled_features, labeled_preds, test_feature, kmax, A)
        simi = simi_matrix[count_idx]
        sorted_idx = simi.argsort()[::-1]
        labels = []
        for idx in sorted_idx:
            if count_idx == idx:
                continue
            key = str(count_idx) + '-' + str(idx)
            if key not in pairs_dict.keys():
                continue
            pairs = pairs_dict[key]
            pairs = [(p[0], p[1]) for p in pairs]
            edges += pairs
            labels.append(idx)
            if len(labels) >= topk:
                break
    print("******************* final *******************")
    return construct_csr_matrix_by_edge(edges, all_features.shape[0])

def propagation_base_on_graph(dataset, outputs, A, buffer_path, alpha_1=0.5, alpha_2=1, beta=0):
    all_features, all_preds, whole_to_invdual, idvdual_to_whole, \
        labeled_frames = get_features(dataset, outputs)
    all_features = feature_norm(all_features)
    all_preds = softmax(all_preds, axis=1)
    for labeled_frame in labeled_frames:
        idx = int(labeled_frame['idx'])
        labels = labeled_frame['label']
        pred = np.zeros(all_preds.shape[1])
        pred[labels[0] + 1] = 1
        all_preds[idx] = pred

    print("total features:", len(all_features))
    # construction
    dir = 'buffer/' + dataset.dataset_name + '/' + str(dataset.active_ratio)
    check_dir(dir)
    path = os.path.join(dir, 'affinity_matrix.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            affinity_matrix = pickle.load(f)
    else: 
        affinity_matrix = graph_construction(dataset, all_features, idvdual_to_whole, labeled_frames, all_preds, A, buffer_path)
        with open(path, 'wb') as f:
            pickle.dump(affinity_matrix, f)
    class_num = len(dataset.classlist)

    inds = dataset.get_trainidx()
    start_id = 0
    labeled_start_id = 0
    center_map = []
    for idx in inds:
        labeled_idx = []
        feat = dataset.get_feature(idx)
        end = start_id + feat.shape[0]
        while labeled_start_id < len(labeled_frames) and start_id <= labeled_frames[labeled_start_id]['idx'] < end:
            labeled_idx.append(labeled_frames[labeled_start_id]['idx'])
            labeled_start_id += 1

        if len(labeled_idx) == 0:
            index = np.ones(feat.shape[0]) * -1
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

    train_y = []
    # get score and normalize
    scores = []
    for trainidx in dataset.get_trainidx():
        labels = dataset.get_init_frame_label(trainidx)
        for labeli in range(len(labels)):
            scores.append(outputs[trainidx][2][labeli])
    scores = np.array(scores)
    scores = softmax(scores, axis=1)
    # 归一化softmax

    begin = 0
    for trainidx in dataset.get_trainidx():
        labels = dataset.get_init_frame_label(trainidx)
        for labeli in range(len(labels)):
            label = labels[labeli]
            trainys = np.zeros(class_num + 1)
            # background = 0
            if len(label) > 0:
                trainys[label[0] + 1] = 1
                # background = 0
            else:
                trainys = scores[begin + labeli]
            train_y.append(trainys)
        begin += len(labels)
    train_y = np.array(train_y)

    label_distributions, unnorm_dist = graph_propagation(affinity_matrix, train_y, center_map, 
        normalized=True, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta)
    begin = 0
    for trainidx in dataset.get_trainidx():
        frame_cnt = len(outputs[trainidx][2])
        outputs[trainidx][2] = label_distributions[begin:begin+frame_cnt]
        begin += frame_cnt
    
    # maxcount = 500
    # for idx, output in outputs.items():
    #     frame_labels = dataset.get_gt_frame_label(idx)
    #     for frameid in range(len(output[1])):
    #         if np.max(label_distributions[begin+frameid])>threshold and len(output[1][frameid])==0:
    #             label = np.argmax(label_distributions[begin+frameid])
    #             output[1][frameid].append(label)
    #             all += 1
    #             if len(frame_labels[frameid])>0 and frame_labels[frameid][0]==label:
    #                 correct += 1
        #         if all == maxcount:
        #             break
        # if all == maxcount:
        #             break
        # begin += len(output[1])
    # print("all added label:", all, "correct label:", correct)

def store_similarity(dataset, all_features, all_preds, idvdual_to_whole, labeled_frames, A):
    simi = np.zeros((len(labeled_frames), len(labeled_frames)))
    pairs_dict = {}
    kmax = 7

    frame_labels = dataset.frame_labels
    for labeled_frame in labeled_frames:
        idx = labeled_frame['idx']
        videoid = int(labeled_frame['id'].split('-')[0])
        frameid = int(labeled_frame['id'].split('-')[1])
        frames = frame_labels[videoid]
        labeled_lb, labeled_rb, labeled_is_active, _ = acquire_boundaries(dataset, \
            videoid, frameid)
        labeled_frame['left'] = labeled_lb
        labeled_frame['right'] = labeled_rb
        labeled_frame['active'] = labeled_is_active
    
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

    dir = 'buffer/' + dataset.dataset_name + '/' + str(dataset.active_ratio)
    check_dir(dir)
    path = os.path.join(dir, 'simi_matrix.npy')
    np.save(path, simi)
    path = os.path.join(dir, 'pairs.json')
    with open(path, 'w') as f:
        res = json.dumps(pairs_dict)
        f.write(res)
    return simi, pairs_dict
