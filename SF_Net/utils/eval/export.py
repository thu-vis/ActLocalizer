import os
import torch
import json
# import utils
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
from scipy.special import softmax
from collections import defaultdict
from torch.autograd import Variable
from .classificationMAP import getClassificationMAP as cmAP
from .eval_detection import ANETdetection
from .eval_frame import FrameDetection
from .eval_utils import get_true_and_false
from ..kNN_utils import get_features, construct_csr_matrix_by_edge
from ..utils import check_dir, pickle_save_data, pickle_load_data
from ..train import act_expand
import pickle
from scipy import io as sio
import warnings

try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.enabled = False
except:
    None

def export_data(iter,
            dataset,
            args,
            model,
            device,
            fps=25,
            stride=16,
            save_matrix=False):
    '''
    '''
    export_dir = os.path.join(args.dir, "export_data")
    check_dir(export_dir)
    classlist = dataset.get_classlist()
    centers = [[] for _ in range(len(classlist))]
    inds = dataset.get_trainidx()
    outputs = {}
    tcams = []
    scores = []
    for idx in inds:
        feat = dataset.get_feature(idx)
        feat = torch.from_numpy(np.expand_dims(feat,
                                               axis=0)).float().to(device)
        cur_label = dataset.get_single_label(idx) # cur_label is single frame labels
        with torch.no_grad():
            _, logits_f, _, logits_r, tcam, _, _, att_logits = model(
                Variable(feat), device, is_training=False)
            tcam = tcam.data.cpu().numpy().squeeze()
            # if args.background:
            #     tcam = tcam[:, 1:]
            assert len(cur_label) == len(tcam)
            tcams.append(tcam)
            score = att_logits.cpu().data.numpy().squeeze()
            scores.append(score)

            for jdx, ls in enumerate(cur_label):
                if len(ls) > 0:
                    for l in ls:
                        centers[l].append(tcam[jdx])
                # if dataset.is_actively_labeled(idx, jdx):
                #     cur_label[jdx] = []
            # outputs += [[idx, cur_label, tcam]]
            outputs[idx] = [idx, cur_label, tcam]

    all_features, all_preds, whole_to_invdual, idvdual_to_whole, \
        labeled_frames = get_features(dataset, outputs)

    all_actions = dataset.all_actions
    classlist = dataset.classlist
    for action in all_actions:
        videoid = action[0]
        start = action[1] * dataset.fps / dataset.stride
        end = action[2] * dataset.fps / dataset.stride
        if end < start:
            continue
        elif end - start < 1.0:
            end += 1
        start = max(0, int(start))
        frame_label = dataset.active_labels[videoid]
        max_len = len(frame_label)
        end = min(max_len, int(end))
        single = action[4]
        for item in single:
            id = int(item * dataset.fps / dataset.stride)
            for labeled_frame in labeled_frames:
                if str(videoid) + '-' + str(id) == labeled_frame['id']:
                    if 'lgt' in labeled_frame.keys():
                        labeled_frame['lgt'].append(start)
                        labeled_frame['rgt'].append(end - 1)
                    else:
                        labeled_frame['lgt'] = [start]
                        labeled_frame['rgt'] = [end - 1]

    act_expand(args,
        dataset,
        model,
        device,
        centers=None,
        prop=args.prop)
    res = {
        "gt_labels": dataset.all_frame_labels,
        "single_frames": labeled_frames,
        'single_frame_labels': dataset.single_frame_labels,
        "pseudo_labels": dataset.frame_labels,
        "whole_to_indvdual": whole_to_invdual,
        "indvdual_to_whole": idvdual_to_whole,
        "trainidx": dataset.get_trainidx(),
        "testidx": dataset.get_testidx(),
        "videonames": dataset.videoname,
        "classlist": classlist,
        "args": args
    }
    dir = 'buffer/' + dataset.dataset_name + '/' + str(dataset.active_ratio)
    path = os.path.join(dir, 'pairs.json')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            pairs_dict = json.load(f)

        edges= []

        for key in pairs_dict.keys():
            pairs = pairs_dict[key]
            pairs = [(p[0], p[1]) for p in pairs]
            edges += pairs
        print("******************* final *******************")
        affinity_matrix = construct_csr_matrix_by_edge(edges, all_features.shape[0])
        sio.mmwrite(os.path.join(export_dir, "affinity_matrix.mtx"), affinity_matrix)
    else:
        warnings.warn('pairs not found!')
    tcams = np.concatenate(tcams)
    scores = np.concatenate(scores)
    np.save(os.path.join(export_dir, "all_features.npy"), all_features)
    np.save(os.path.join(export_dir, "classification_prediction.npy"), tcams)
    np.save(os.path.join(export_dir, "action_prediction.npy"), scores)
    pickle_save_data(os.path.join(export_dir, "meta_data.pkl"), res)

    if args.prop:
        None
        # save affinity_matrix
