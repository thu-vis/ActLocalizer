import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csgraph
from time import time, sleep
from tqdm import tqdm
import warnings
from scipy.optimize import minimize
from scipy.stats import entropy
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import euclidean_distances, paired_distances
from .utils import feature_norm
from .dataset.video_dataset import left_active, right_active
import copy

def get_features(dataset, outputs, use_test=False):    
    inds = dataset.get_trainidx()
    if use_test is True:
        inds = dataset.get_testidx()
    classlist = dataset.get_classlist()
    all_features = []
    idvdual_to_whole = {}
    whole_to_invdual = {}
    start_id = 0
    labeled_frames = []
    all_gt_labels = {}
    all_preds = []
    for idx in inds:
        feat = dataset.get_feature(idx)
        frame_label = dataset.get_init_frame_label(idx)
        gt_frame_labels = dataset.get_gt_frame_label(idx)
        video_label = dataset.get_video_label(idx)
        if len(video_label) == 0:
            continue
        all_gt_labels[idx] = gt_frame_labels
        all_features.append(feat)
        all_preds.append(outputs[idx][2])
        for i in range(feat.shape[0]):
            frame_id = str(idx) + "-" + str(i)
            frame_idx_in_whole = start_id + i
            idvdual_to_whole[frame_id] = frame_idx_in_whole
            whole_to_invdual[frame_idx_in_whole] = frame_id
            if len(frame_label[i]) > 0:
                labeled_frames.append({
                    "id": frame_id,
                    "idx": frame_idx_in_whole,
                    "label": frame_label[i]
                })
        start_id = start_id + feat.shape[0]
    print("frame_count: ", len(all_features))
    all_features = np.concatenate(all_features, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_features, all_preds, whole_to_invdual, idvdual_to_whole, labeled_frames

def left_bound(frame_label, frame_idx, label):
    assert label in frame_label[frame_idx]
    start = frame_idx - 1
    if start < 0:
        return 0
    for idx in range(start, -1, -1):
        if len(frame_label[idx]) > 0:
            return idx + 1
    return 0

def right_bound(frame_label, frame_idx, label):
    assert label in frame_label[frame_idx]
    max_idx = len(frame_label) - 1
    start = frame_idx + 1
    if start > max_idx:
        return max_idx
    for idx in range(start, max_idx + 1):
        if len(frame_label[idx]) > 0:
            return idx - 1
    return max_idx

def acquire_boundaries(dataset, video_id, frame_id):
    active_label = dataset.is_actively_labeled(video_id, frame_id)
    if active_label:
        label = dataset.active_labels[video_id][frame_id][0]
        lb = left_active(dataset.active_labels[video_id], frame_id, label)
        rb = right_active(dataset.active_labels[video_id], frame_id, label)
        is_active = True
    else:
        label = dataset.single_frame_labels[video_id][frame_id][0]
        lb = left_bound(dataset.frame_labels[video_id], frame_id, label)
        rb = right_bound(dataset.frame_labels[video_id], frame_id, label)
        is_active = False
    return lb, rb, is_active, label

def encoding_pair(i, j):
    return str(i) + "-" + str(j)

def decoding_pair(pair):
    a, b = pair.split("-")
    return int(a), int(b)

def construct_csr_matrix_by_edge(edges, n):
    res = [[] for _ in range(n)]
    for e in edges:
        res[e[0]].append(e[1])
        res[e[1]].append(e[0])
    res = [list(set(item)) for item in res]
    indptr = [0]
    indices = []
    for r in res:
        indptr.append(indptr[-1] + len(r))
        indices += r
    data = np.array(indices)
    data = (data * 0 + 1.0).tolist()
    affinity_matrix = sparse.csr_matrix((data, indices, indptr), \
        shape=(n, n))
    return affinity_matrix

def sigma_func(x, tau=0.1): # tau = 4 in the original paper
    x = np.abs(x)
    x[x>tau] = tau
    return x

def sigma_prime_func(x, tau=0.1):
    pos = (x > 0) & (x < tau)
    neg = (x < 0) & (x > -tau)
    x = x * 0
    x[pos] = 1
    x[neg] = -1
    return x

def build_laplacian_graph(affinity_matrix):
    instance_num = affinity_matrix.shape[0]
    laplacian = csgraph.laplacian(affinity_matrix, normed=True)
    laplacian = -laplacian
    if sparse.isspmatrix(laplacian):
        diag_mask = (laplacian.row == laplacian.col)
        laplacian.data[diag_mask] = 0.0
    else:
        laplacian.flat[::instance_num + 1] = 0.0  # set diag to 0.0
    return laplacian

def graph_propagation(graph_matrix, train_y, center_map, alpha_1=0.5, alpha_2=1, beta=0, \
    max_iter=20, tol=1e-12, normalized=False, confident_unlabeled=None, special_labeled=None, tau_changed=[]):
    # alpha1: unlabeled alpha
    # alpha2: labeled alpha
    t0 = time()

    if sparse.isspmatrix(graph_matrix):
        graph_matrix = graph_matrix.tocsr()

    # W = copy.deepcopy(graph_matrix)
    # row, col = W.nonzero()
    # for i in range(len(row)): 
    #     W[row[i], col[i]] = 1
    # row, col = W.nonzero()
    # N = W.shape[0]
    # d = np.zeros(N)
    # for i in range(len(row)):
    #     d[row[i]] += 1
    #     d[col[i]] += 1

    graph_matrix = build_laplacian_graph(graph_matrix)

    y = np.array(train_y)
    n_samples, n_classes = y.shape
    classes = np.array(range(n_classes))
    l_previous = np.zeros((n_samples, n_classes))

    unlabeled = y.max(axis=1) < 1
    labeled = y.max(axis=1) >= 1
    identify_matrix = np.ones_like(unlabeled).astype(float)
    identify_matrix = sparse.diags(identify_matrix, format="csr")
    Gamma_vector = np.zeros_like(unlabeled, dtype=float)
    Gamma_vector[unlabeled] = alpha_1
    Gamma_vector[labeled] = alpha_2
    if confident_unlabeled is not None:
        for id in confident_unlabeled:
            if unlabeled[id]:
                Gamma_vector[id] = 0

    if special_labeled is not None:
        for id in special_labeled:
            if labeled[id]:
                Gamma_vector[id] = 4

    # alpha_vector = copy.deepcopy(Gamma_vector)

    Gamma_vector = 1 / (1 + Gamma_vector)
    Gamma = sparse.diags(Gamma_vector, format="csr")
    B = np.ones_like(unlabeled).astype(float) * beta
    uncentered = center_map == -1
    B[uncentered] = 0
    B =  B / (1 + Gamma_vector)
    B = sparse.diags(B, format="csr")
    one_minus_Gamma = identify_matrix - Gamma
    y_static = safe_sparse_dot(one_minus_Gamma, y)
    Gamma_S = safe_sparse_dot(Gamma, graph_matrix)
    print("preprocess time:", time() - t0)

    # initialize distributions
    label_distributions_ = y

    # y_static_labeled = np.copy(label_distributions_)
    # y_static = y_static_labeled * (1 - alpha_1)

    n_iter_ = 1
    output = 25620 - 6
    label = 10 + 1
    a=Gamma_S[output].nonzero()[1]
    from IPython import embed
    # embed()
    # losses = []
    # output = tau_changed[-1]
    tau = 8
    for _ in range(max_iter):
        if np.abs(label_distributions_ - l_previous).sum() < tol:
            break
        l_previous = label_distributions_.copy()
        centers = label_distributions_[center_map]
        gap = label_distributions_ - centers
        third_term_a = sigma_func(gap)
        third_term_b = sigma_prime_func(gap)

        for num in tau_changed:
            labeled_gap = gap[num]
            labeled_third_term_a = sigma_func(labeled_gap, tau=tau)
            labeled_third_term_b = sigma_prime_func(labeled_gap, tau=tau)
            third_term_a[num] = labeled_third_term_a
            third_term_b[num] = labeled_third_term_b

        third_term = safe_sparse_dot(B, third_term_a*third_term_b)
        label_distributions_ = safe_sparse_dot(Gamma_S, label_distributions_)
        # normalizer = np.sum(label_distributions_, axis=1)[:, np.newaxis]
        # normalizer = normalizer + 1e-20
        # label_distributions_ /= normalizer
        # print(label_distributions_[output][0], label_distributions_[output][label])
        # print(y_static[output][0], y_static[output][label])
        # print(third_term[output][0], third_term[output][label])
        
        # from IPython import embed
        # embed();exit()
        label_distributions_ = label_distributions_ + y_static - third_term
        # print(label_distributions_[center_map[output]][0], label_distributions_[center_map[output]][11])
        # print(label_distributions_[output][0], label_distributions_[output][label])
        # print('*******************************************')
        # label_distributions_ = (
        #             np.multiply(alpha_1, label_distributions_) + y_static
        #         )
        n_iter_ += 1

        # loss = 0
        # for i in range(len(row)):
        #     first = label_distributions_[row[i]] / d[row[i]] - label_distributions_[col[i]] / d[col[i]]
        #     loss += np.sum(first * first)
        # gap = label_distributions_ - y
        # gap = gap * gap
        # gap = gap.sum(axis=1)
        # loss += np.dot(alpha_vector, gap)
        # loss += beta * np.sum(third_term_a * third_term_a)
        # losses.append(loss)

    else:
        warnings.warn(
            'max_iter=%d was reached without convergence.' % max_iter,
            category=ConvergenceWarning
        )
    
    label_distributions_ = np.maximum(label_distributions_, 0)

    unnorm_dist = label_distributions_.copy()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # x = list(range(1, max_iter + 1))
    # plt.plot(x, losses)
    # plt.savefig('1.jpg')
    a = Gamma_S[output].nonzero()[1]
    from IPython import embed
    # embed()

    if normalized:
        normalizer = np.sum(label_distributions_, axis=1)[:, np.newaxis]
        normalizer = normalizer + 1e-20
        label_distributions_ /= normalizer

    print("graph propagation time:", time() - t0)

    return label_distributions_, unnorm_dist
 