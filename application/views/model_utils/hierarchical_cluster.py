"""
This file include classes for alignements clustering using KMedoids.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn_extra.cluster import KMedoids
from IPython import embed
from copy import deepcopy

from .reorder import get_adjancent_dists, get_order_and_dists_adj, get_detail_order
from .y_layout import solve_y_layout_adj
from .alignment import Alignment, Line
from .frame_selector import FrameSelector, LineFrameSelector
from sklearn.metrics import pairwise_distances

# >>> Cluster by k-medoids
class KMedoidsCluster:
    """
    do cluster
    """
    def __init__(self, class_name, dist_matrix):
        """
        init alignments and align matrix

        Parameters:
        n -  total number of videos in this action
        actions - list of actions
        helper - cluster helper
        """
        # clustering meta infomation
        self.class_name = class_name
        self.n_clusters = dist_matrix.shape[0]
        assert dist_matrix.shape == (self.n_clusters, self.n_clusters)
        self.dists_matrix = dist_matrix

        self.size_limit = 7
        self.zk_delta = 0.2

    def do_cluster(self):
        start_idx = list(range(self.n_clusters))
        result = self.do_cluster_in_subset(start_idx)
        return result

    def do_cluster_in_subset(self, idx):
        # 0. get idx dist
        dist = self.dists_matrix[idx,:][:,idx]
        
        # 1. find best fit bandwidth
        lmax = min(7, len(idx))
        lmin = min(5, len(idx))
        lns = list(range(lmin, lmax + 1))
        cls, zks = [], []

        for i in lns:
            kmlabels = KMedoids(n_clusters=i, metric='precomputed',
                                method='pam', init='heuristic').fit(dist).labels_
            result = []
            for _ in range(i):
                result.append([])
            for i in range(dist.shape[0]):  
                result[kmlabels[i]].append(idx[i])

            cls.append(result)
            # zk = silhouette_score(dist, kmlabels, metric='precomputed')
            zk = self._calculate_zk(result)
            zks.append(zk)

        # plt.plot(lns, zks)
        # plt.show()
        # import IPython;IPython.embed();
        # exit()

        if lmax == 2:
            max_index = 0
        else:
            max_index = np.array(zks).argmax()
            if max_index == 0 and lmin == 2:
                zk = zks[0]
                zks[0] = -1
                max_index2 = np.array(zks).argmax()
                if max_index2 != 0 and zk - zks[max_index2] <= self.zk_delta:
                    max_index = max_index2

        clusters = cls[max_index]

        # 2. solve node and large cluster

        if len(clusters) == 1:
            return clusters[0]

        result = []
        for cluster in clusters:
            if len(cluster) > self.size_limit:
                _node = self.do_cluster_in_subset(cluster)
            else:
                _node = cluster
            result.append(_node)

        return result

    def _calculate_zk(self, clusters):
        """
        calculate silhouette_score for clusters
        """
        sum_zk = 0
        count = 0
        n_clusters = len(clusters)
        for i in range(n_clusters):
            ids = clusters[i]
            length = len(ids)

            if length > 1:
                # calculate u(s)
                alpha = length / (length - 1)
                us = np.mean(self.dists_matrix[ids][:, ids], axis=1) * alpha

                if len(clusters) > 1:
                    # calculate v(s)
                    vs = 10000  * np.ones(length)
                    for j in range(n_clusters):
                        if i == j:
                            continue
                        _ids = clusters[j]
                        _vs = np.mean(self.dists_matrix[ids][:, _ids], axis=1)
                        vs = np.array([vs, _vs]).min(axis=0)
                else:
                    vs = np.zeros(length)

                # calculate max(u,v)
                uv_max = np.array([us, vs]).max(axis=0)

                # finally get z(s)
                zs = (vs - us) / uv_max
                sum_zk += zs.sum()

            count += length

        return sum_zk / count

# >>> Utils for layout
def uniform_align(idx_dict, ac):
    """
    idx_dict: dict idx->[pos, length]
    ac: result of alignment clustering
    """
    ac_list = ac.alignments[0].ids
    ret_aligns = np.zeros(
        (len(idx_dict), ac.alignments[0].frames.shape[1]), dtype='int32')

    i, ln = 0, len(ac_list)
    while i < ln:
        idx = ac_list[i]
        pos, length = idx_dict[idx]
        start, final = i, i + length
        ret_aligns[pos] = ac.alignments[0].frames[start: final].mean(
            axis=0)
        i = final

    return ret_aligns.tolist()
    
def interpolate_dist(align, adj_dist, node):
    if len(adj_dist) == 0:
        # leaf node
        return [], [align[:]]
    
    adj_dist = np.array(adj_dist).transpose()
    aligns = np.array(node['aligns']).transpose()
    n_col, n_row = len(align), adj_dist.shape[1]
    ret = np.full((n_col, n_row), -1)
    ret_aligns = np.full((n_col, n_row + 1), -1)
    assert n_col >= adj_dist.shape[0]
    
    # 1. Add value of distance
    j = 0
    for i in range(n_col):
        if align[i] > -1:
            ret[i] = adj_dist[j]
            ret_aligns[i] = aligns[j]
            j += 1
    assert j == adj_dist.shape[0] and j == aligns.shape[0]

    # 2. Do interopolation for dist
    left = np.full((n_col,), -1)
    right = np.full((n_col,), -1)
    for i in range(n_col):
        if align[i] != -1:
            left[i] = i
        elif i > 0:
            left[i] = left[i-1]
        _i = n_col - 1 - i
        if align[_i] != -1:
            right[_i] = _i
        elif _i < n_col - 1:
            right[_i] = right[_i+1]
    
    for i in range(n_col):
        if align[i] == -1:
            value = 0
            bound = 0
            if left[i] != -1 and right[i] != -1:
                value += (i - left[i]) * ret[right[i]]
                value += (right[i] - i) * ret[left[i]]
                bound += (right[i] - left[i])
            else:
                if left[i] != -1:
                    value += ret[left[i]]
                if right[i] != -1:
                    value += ret[right[i]]
                bound = 1
            assert bound > 0
            ret[i] = value / bound
    
    ret = ret.transpose()
    ret_aligns = ret_aligns.transpose()
    return ret.tolist(), ret_aligns.tolist()

def interpolate_orders(align, detail_orders):
    ret = []
    length = len(align)
    lf, rt = [0] * length, [0] * length
    cur_orders = []
    for i in range(length):
        if align[i] > 0:
            lf[i] = 0
        else:
            lf[i] = lf[i - 1] + 1 if i > 0 else 10000
        r = length - i - 1
        if align[r] > 0:
            rt[r] = 0
        else:
            rt[r] = rt[r + 1] + 1 if i > 0 else 10000
    cur = 0
    for i in range(length):
        if align[i] > 0:
            cur_orders.append(detail_orders[cur])
            cur += 1
        else:
            cur_orders.append([])

    for i in range(length):
        if lf[i] < rt[i]:
            ret.append(deepcopy(cur_orders[i - lf[i]]))
        else:
            ret.append(deepcopy(cur_orders[i + rt[i]]))
    return ret


# >>> Generate hierarchy by cluster
def decode_action_id(id):
    values = id.split('-')
    return int(values[0]), int(values[1])

def check_center_index(aligns, children):
    """
    check center index for aligns, return not all -1 lines
    """
    center_aligns = deepcopy(aligns)
    for i, align in enumerate(center_aligns):
        center_pos = children[i]['center_align']
        index = 0
        for j in range(len(align)):
            if align[j] != -1:
                if center_pos[index] == -1:
                    align[j] = -1
                index += 1
    cols = [col_index for indices in np.argwhere(np.sum(np.array(center_aligns) + 1, axis=0)!=0)\
                        for col_index in indices]
    return cols, np.array(center_aligns)

def check_pairs_and_get_scores(line, aligns, lines, afmatrix):
    """
    check every frame pos lambda as penalization
    """
    aligns_T = np.array(aligns).transpose()
    width, size = aligns_T.shape
    pair_numbers = np.zeros((width, ), dtype=np.int32)

    indexes = [0] * size
    for i in range(width):
        align = aligns_T[i]
        pair_num = 0
        for j in range(size):
            if align[j] != -1:
                pair_num += lines[j].pair_numbers[indexes[j]]
                for _j in range(j):
                    if align[_j] != -1 and \
                        (lines[j].index + indexes[j]) in afmatrix[lines[_j].index + indexes[_j]]:
                        pair_num += lines[j].size * lines[_j].size
        for j in range(size):
            if align[j] != -1:
                indexes[j] += 1
        pair_numbers[i] = pair_num
    line.pair_numbers = pair_numbers

    if line.size == 1:
        return [1] * width
    Cn2 = line.size * (line.size - 1) / 2
    lambdas = 1 - pair_numbers.astype(np.float64) / Cn2
    return lambdas.tolist()

def get_rep_cols_images(rep_cols, aligns, center_aligns, lines):
    aligns_T = np.array(aligns).transpose()
    calign_T = center_aligns.transpose()
    ret = []
    w, n = aligns_T.shape
    major_row = np.argmax(np.sum(calign_T + 1, axis=0))
    indexes = [-1] * n
    for i in range(w):
        align = aligns_T[i]
        for j in range(n):
            if align[j] > -1:
                indexes[j] += 1
        if i in rep_cols:
            if calign_T[i][major_row] > -1:
                ret.append(int(major_row))
            else:
                rows = list(np.nonzero(calign_T[i] + 1)[0])
                features = []
                for row in rows:
                    features.append(lines[row].features[indexes[row]])
                dist = pairwise_distances(np.array(features), metric='euclidean')
                ret.append(int(rows[dist.sum(axis=0).argmin()]))
    return ret
        

def get_hierarchy_by_cluster(cdx, raw_clusters, actions, helper, dists, use_pre_order=False):
    """
    get hierarchy from cluster
    Params:
        cdx - class index
        raw_clusters - cluster result
        actions - cluster action list
        helper - cluster helper for args
    Retures:
        tree - hierarchy of cluster
        all_lines - aligns of every node
    """
    middle_node_index = 0
    affinity_matrix = helper.affinity_matrix
    selector = LineFrameSelector(cdx, helper)
    # rep_frame_num = 8
    # all_lines, all_adj_dists = {}, {}
    all_orders, all_centers = {}, {}
    if use_pre_order:
        all_orders = helper.all_orders
        all_centers = helper.all_centers
    remove_list = set(helper.remove_list)

    def get_tree(raw):
        nonlocal middle_node_index, affinity_matrix
        """
        get tree of raw list
        Return:
            node - tree node
            adj_dists - adjancant distances of node
            line - merged line of tree
            items - total item of cluster
        """
        if type(raw) != list or len(raw) == 1:
            # >> leaf node info 
            node = {}
            if type(raw) == list:
                raw = raw[0]
            action = actions[raw]

            # 1. basic infomations
            node['name'] = action['id']
            node['bound'] = [action['left_bound'], action['right_bound']]
            node['idx'] = action['idx']
            node['video_id'] = action['video_id']
            node['single_frame'] = action['single_frame']
            node['pred_scores'] = action['pred_scores']
            if action['id'] in remove_list:
                return None, [], None, [], [], []

            # 2. Add action keyframes for leaf nodes
            node['key_frames'] = helper.key_frames[node['name']]

            # 3. get line of leaf action
            bound = action['bound']
            line = Line(
                bound, action['idx'], helper.features[bound[0]:bound[1]+1])

            # 4. set center itself
            node['center'] = node['name']
            node['center_align'] = [1] * len(node['pred_scores'])
            node['center_length'] = len(node['center_align'])

            rep_cols = selector.select_frames(line, list(range(node['center_length'])), 12)
            node["rep_cols"] = rep_cols
            node["rep_rows"] = [0] * len(rep_cols)

            # 5. add node in align objects
            # all_lines[node['name']] = line.copy()
            # all_adj_dists[node['name']] = []

            # 6. detail order
            detail_orders = []
            for i in range(len(node['center_align'])):
                detail_orders.append([0])
            return node, [], line, [raw], detail_orders, []

        # >> middle node info
        node = {}
        children, adj_dists, lines, clusters, items = [], [], [], [], []
        all_detail_orders, v_adj_dists = [], []
        idx_dict, idx_seq = {}, 0
        recal_center = False
        remove_indexes = []
        
        for id, cluster in enumerate(raw):
            _node, _dist, _line, _item, _detail_orders, _v_dist = get_tree(cluster)
            if _node == None:
                recal_center = True
                remove_indexes.append(id)
                continue
            children.append(_node)
            adj_dists.append(_dist)
            lines.append(_line)
            clusters.append(_item)
            all_detail_orders.append(_detail_orders)
            v_adj_dists.append(_v_dist)

            idx_dict[_line.ids[0]] = [idx_seq, len(_line.ids)]
            idx_seq += 1

        node['name'] = "M{}-{}".format(cdx, middle_node_index)
        middle_node_index += 1
        if len(children) == 0:
            return None, [], None, [], [], []

        # 1. Alignment
        pause = False
        # if raw == raw_clusters:
        #     pause = True
        ac = Alignment(0, None, helper, pre_aligned=True, alignments=lines, pause=pause)
        ac.cluster_to(1)
        aligns = uniform_align(idx_dict, ac)
        aligns_clear = []
        for align in aligns:
            aligns_clear.append(list(map(lambda x: 1 if x != -1 else -1, align)))
        line = ac.alignments[0]
        mismatch_scores = check_pairs_and_get_scores(line, aligns, lines, affinity_matrix)
        
        # 2. Reordering
        if not use_pre_order:
            case = -1
            if raw == raw_clusters:
                case = cdx
            reorder, adj_dist = get_order_and_dists_adj(lines, aligns, affinity_matrix, case=case)
        else:
            reorder = all_orders[node['name']]
            if recal_center:
                _reorder = []
                for i in reorder:
                    if i in remove_indexes:
                        continue
                    _reorder.append(i)
                reorder = [0] * len(_reorder)
                argsort = np.argsort(_reorder)
                for i in range(len(_reorder)):
                    reorder[argsort[i]] = i
                
        tmp_lines, tmp_aligns = [], []
        for i in reorder:
            tmp_lines.append(lines[i])
            tmp_aligns.append(aligns_clear[i])
        if use_pre_order:
            adj_dist = get_adjancent_dists(tmp_lines, tmp_aligns, affinity_matrix)
        adj_dist = adj_dist.tolist()

        # 2.5 Reordering for detail
        detail_orders, v_dist = get_detail_order(tmp_lines, tmp_aligns, affinity_matrix)

        # 3. Layout - y
        total_adj_dist, total_aligns, labels = [], [], []
        total_adj_dist.append([0] * len(aligns_clear[0]))
        total_orders, cur_index = [], 0
        line_nums = [1]
        for i in range(len(reorder)):
            index = reorder[i]
            # node_adj_dist, node_aligns = interpolate_dist(aligns_clear[index], adj_dists[index], children[index])
            node_adj_dist, node_aligns = interpolate_dist(aligns_clear[index], v_adj_dists[index], children[index])
            node_detail_orders = interpolate_orders(aligns_clear[index], all_detail_orders[index])
            node_detail_orders = np.array(node_detail_orders) + cur_index
            cur_index += node_detail_orders.shape[1]
            total_orders.append(node_detail_orders)
            if i > 0:
                total_adj_dist.append(adj_dist[i-1])
                line_nums.append(1)
            total_adj_dist.extend(node_adj_dist)
            total_aligns.extend(node_aligns)
            labels.extend([i] * len(node_aligns))
            line_nums[-1] += len(node_adj_dist)
        total_adj_dist = np.array(total_adj_dist)
        total_aligns = np.array(total_aligns)
        total_orders = np.hstack(total_orders)
        # TODO change layout using total_orders
        y_layout = solve_y_layout_adj(total_adj_dist, total_aligns, labels, total_orders).round(4)
        
        # straighten
        assert y_layout.shape[0] == np.sum(line_nums)
        start, end = 0, 0
        for line_num in line_nums:
            start = end
            end = start + line_num
            if line_num == 1:
                y_layout[start] = y_layout[start][0]
                continue
            local_aligns = total_aligns[start: end].transpose() + 1
            align_length = local_aligns.shape[0]

            # left search
            lf_start = 0
            while lf_start < align_length:
                if np.sum(local_aligns[lf_start]) > 0:
                    break
                lf_start += 1
            lf_end = lf_start
            single_line = -1
            while lf_end < align_length:
                value_cols = np.nonzero(local_aligns[lf_end])[0]
                if single_line == -1:
                    single_line = value_cols[0]
                if len(value_cols) > 1 or \
                    (len(value_cols) == 1 and value_cols[0] != single_line):
                    break
                lf_end += 1
            if lf_start != lf_end:
                single_line = single_line + start
                for i in range(lf_start, lf_end):
                    y_layout[single_line, i] = y_layout[single_line, lf_end]
            
            # right search
            rt_start = align_length - 1
            while rt_start > -1:
                if np.sum(local_aligns[rt_start]) > 0:
                    break
                rt_start -= 1
            rt_end = rt_start
            single_line = -1
            while rt_end > -1:
                value_cols = np.nonzero(local_aligns[rt_end])[0]
                if single_line == -1:
                    single_line = value_cols[0]
                if len(value_cols) > 1 or \
                    (len(value_cols) == 1 and value_cols[0] != single_line):
                    break
                rt_end -= 1
            if rt_start != rt_end:
                single_line = single_line + start
                for i in range(rt_start, rt_end, -1):
                    y_layout[single_line, i] = y_layout[single_line, rt_end]

        # if raw == raw_clusters:
        #     embed();exit()
        
        # 4. add node info
        node['children'], node['aligns'] = [], []
        _clusters = []
        for index in reorder:
            node['children'].append(children[index])
            node['aligns'].append(aligns_clear[index])
            _clusters.append(clusters[index])
            items.extend(clusters[index])
        node['y_layout'] = y_layout.tolist()
        node['mismatch_scores'] = mismatch_scores

        # 5. calculate node center
        if not use_pre_order or recal_center:
            # 1. distances center
            n_clusters = len(_clusters)
            clus_dists = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                for j in range(i):
                    clus_dists[i][j] = dists[_clusters[i], :][:, _clusters[j]].mean()
            clus_dists = clus_dists + clus_dists.T
            # center_index = int(clus_dists.sum(axis=1).argmin())
            
            # 2. longest length center
            lns = []
            for child in node['children']:
                lns.append(child['center_length'])
            # center_index = int(np.array(lns).argmax())

            # 3. merge 1 and 2
            ln_array = np.array(lns)
            alpha = 0.2
            judge_metric = - clus_dists.sum(axis=1) + alpha * ln_array / ln_array.max() 
            center_index = int(np.array(judge_metric).argmax())

        else:
            center_index = all_centers[node['name']]
    
        node['center'] = node['children'][center_index]['center']
        node['center_index'] = center_index
        node['center_align'] = node['aligns'][center_index][:]
        node['center_length'] = node['children'][center_index]['center_length']
        child_center_align = node['children'][center_index]['center_align']
        child_center_index = 0
        for i in range(len(node['center_align'])):
            if node['center_align'][i] == 1:
                node['center_align'][i] = child_center_align[child_center_index]
                child_center_index += 1

        # 6. select rep lines
        center_cols, center_aligns = check_center_index(node['aligns'], node['children'])
        rep_cols = selector.select_frames(line, center_cols, 12)
        node["rep_cols"] = rep_cols
        node["rep_rows"] = get_rep_cols_images(rep_cols, node['aligns'], center_aligns, tmp_lines)

        # 7. save lines and adj info for update
        # all_lines[node['name']] = line.copy()
        # all_adj_dists[node['name']] = deepcopy(adj_dist)
        if not use_pre_order:
            all_orders[node['name']] = deepcopy(reorder)
            all_centers[node['name']] = center_index
        return node, adj_dist, line, items, detail_orders, v_dist

    node, _, _, _, _, _ = get_tree(raw_clusters)
    save_info = {
        # "all_lines": all_lines,
        # "all_adj_dists": all_adj_dists,
        "all_orders": all_orders,
        "all_centers": all_centers
    }
    return node, save_info




# >>> Load and analyse hierarchy
def get_hierarchy_index(tree):
    """
    get hierarchy index by tree
    Params:
        tree - hierarchy data
    Return:
        node_indexes - dict: name to index
        parent_indexes - dict: name to name
    """
    node_indexes, parent_indexes = {}, {}

    def analyse_tree(node):
        name = node['name']
        node_indexes[name] = node
        if 'children' in node:
            for index, child in enumerate(node['children']):
                parent_indexes[child['name']] = (name, index)
                analyse_tree(child)
        return

    analyse_tree(tree) 
    return node_indexes, parent_indexes

# >>> Update and merge tree and lines
def update_affinity_matrix_by_tree_and_lines(affinity_matrix, tree, all_lines):
    """
    Update affinity matrix by tree and lines
    Params:
        affinity_matrix -  action affinity matrix object
        tree - hierarchy data of action
        all_lines - dict: lines of tree node
    """
    def update_node(node):
        if 'children' not in node:
            return
        
        # calculate merged frames info
        lines = []
        for child in node['children']:
            update_node(child)
            lines.append(all_lines[child['name']])
        aligns = node['aligns']
        merged_frames = merge_affinity_lines(affinity_matrix, lines, aligns)
        
        # append info into affinitymatrix
        merged_line = all_lines[node['name']]
        assert merged_line.width == len(merged_frames)
        affinity_matrix.append_i(merged_frames, merged_line.index)

    update_node(tree)

def merge_affinity_lines(affinity_matrix, lines, aligns):
    """
    merge affinity lines by line info and aligns result
    Params:
        affinity_matrix -  action affinity matrix object
        lines - list: (n,) line info of every line
        aligns - list: (n, w) aligns of every line
    Return:
        frames - (w, ) list of set of merged line
    """
    aligns_T = np.array(aligns).transpose()
    w, n = aligns_T.shape
    assert n == len(lines)

    frames = []
    indexes = [0] * n
    for i in range(w):
        frame = set()
        align = aligns_T[i]
        for j in range(n):
            if align[j] != -1:
                frame = frame | affinity_matrix[lines[j].index + indexes[j]]
                indexes[j] += 1
        frames.append(frame)
    return frames

def clear_aligns(aligns):
    # clear aligns all -1 lines
    align_matrix = np.array(aligns)
    cols = np.nonzero(np.sum(align_matrix + 1, axis=0))
    return align_matrix[:, cols[0]].tolist()

def merge_lines(lines, aligns, afmatrix):
    """
    Merge lines into a new line
        lines - list: (n,) line info of every line
        aligns - list: (n, w) aligns of every line
        afmatrix - affinity_matrix
    """
    aligns_T = np.array(aligns).transpose()
    w, n = aligns_T.shape
    merged_line = Line([0,0], 0, None)
    merged_line.size = 0
    merged_line.ids = []
    merged_line.width = w

    indexes = [0] * n
    frames_r, frames = [], []
    for i in range(n):
        merged_line.size += lines[i].size
        merged_line.ids += lines[i].ids
        frames_r.append(lines[i].frames.transpose())
        frames.append(np.zeros((w, lines[i].size)))
    features = np.zeros((w, lines[0].features.shape[1]))
        
    for i in range(w):
        align = aligns_T[i]
        for j in range(n):
            if align[j] == -1:
                frames[j][i] = -1 * np.ones(lines[j].size)
            else:
                frames[j][i] = frames_r[j][indexes[j]]
                features[i] += lines[j].size * lines[j].features[indexes[j]]
                indexes[j] += 1
    
    merged_line.features = features / merged_line.size
    merged_line.frames = np.hstack(frames).transpose()
    merged_line.index = afmatrix.shape[0]
    aflines = merge_affinity_lines(afmatrix, lines, aligns)
    afmatrix.append(aflines)
    return merged_line

def superpose_aligns(align1: list, aligns2: list):
    """
    Superpose aligns of 2 layers
    number of 1 in align1 should equal w of aligns2
    """
    aligns2_T = np.array(aligns2).transpose()
    ret_T = np.full((len(align1), aligns2_T.shape[1]), -1)
    i2 = 0
    for i in range(len(align1)):
        if align1[i] != -1:
            ret_T[i] = aligns2_T[i2]
            i2 += 1
    assert i2 == aligns2_T.shape[0]
    return ret_T.transpose().tolist()

def relayout_in_target_node(node, indexes, helper):
    """
    Re layout in target node
    Params:
        node - node to do re layout
        indexes - list of changed children indexes in node
        helper - for layout args and afmatrix
    """
    all_lines = helper.all_lines
    all_adj_dists = helper.all_adj_dists
    # all_rep_info = helper.all_rep_info
    affinity_matrix = helper.affinity_matrix

    # 0. merge other lines
    lines = []
    merged_lines, merged_aligns = [], []
    for i, child in enumerate(node['children']):
        if i in indexes:
            lines.append(all_lines[child['name']])
        else:
            merged_lines.append(all_lines[child['name']])
            merged_aligns.append(node['aligns'][i])

    if len(merged_lines) > 0:
        merged_aligns = clear_aligns(merged_aligns)
        merged_line = merge_lines(merged_lines, merged_aligns, affinity_matrix)
        lines.append(merged_line)

    # 1. x layout
    idx_dict, idx_seq = {}, 0
    for line in lines:
        idx_dict[line.ids[0]] = [idx_seq, len(line.ids)]
        idx_seq += 1
    ac = Alignment(0, None, helper, pre_aligned=True, alignments=lines)
    ac.cluster_to(1)
    aligns = uniform_align(idx_dict, ac)
    aligns_clear = []
    for align in aligns:
        aligns_clear.append(list(map(lambda x: 1 if x != -1 else -1, align)))
    if len(merged_lines) > 0:
        merged_aligns = superpose_aligns(aligns_clear[-1], merged_aligns)

    aligns = []
    i1, i2 = 0, 0  # rep align index of changed and other
    for i in range(len(node['children'])):
        if i in indexes:
            aligns.append(aligns_clear[i1])
            i1 += 1
        else:
            aligns.append(merged_aligns[i2])
            i2 += 1

    # 2. calculate distance without Reorder
    lines = list(map(lambda x: all_lines[x['name']], node['children']))
    adj_dist = get_adjancent_dists(lines, aligns, affinity_matrix)

    # 3. Layout - y
    adj_dists = list(map(lambda x: all_adj_dists[x['name']], node['children']))
    total_adj_dist, total_aligns, labels = [], [], []
    total_adj_dist.append([0] * len(aligns[0]))
    for i in range(len(lines)):
        node_adj_dist, node_aligns = interpolate_dist(aligns[i], adj_dists[i], node['children'][i])
        if i > 0:
            total_adj_dist.append(adj_dist[i-1])
        total_adj_dist.extend(node_adj_dist)
        total_aligns.extend(node_aligns)
        labels.extend([i] * len(node_aligns))
    total_adj_dist = np.array(total_adj_dist)
    total_aligns = np.array(total_aligns)
    y_layout = solve_y_layout_adj(total_adj_dist, total_aligns, labels).round(4)

    # 4. save current state
    node['aligns'] = aligns
    node['y_layout'] = y_layout.tolist()
    min_index = node['center_index'] 
    node['center_align'] = node['aligns'][min_index][:]
    child_center_align = node['children'][min_index]['center_align']
    child_center_index = 0
    for i in range(len(node['center_align'])):
        if node['center_align'][i] == 1:
            node['center_align'][i] = child_center_align[child_center_index]
            child_center_index += 1
            
    # update rep pos update
    # rep_info = all_rep_info[node['name']]
    # cur_pos = []
    # line = ac.alignments[0].copy()
    # for i in range(len(rep_info[0])):
    #     idx, pos_in_action = rep_info[0][i], rep_info[1][i]
    #     action_row = line.frames[line.ids.index(idx)]
    #     pos_count = -1
    #     for j in range(action_row.shape[0]):
    #         if action_row[j] > -1:
    #             pos_count += 1
    #             if pos_count == pos_in_action:
    #                 cur_pos.append(j)
    
    # assert len(rep_info[0]) == len(cur_pos)
    # clear_dict = {}
    # for i in range(len(cur_pos)):
    #     clear_dict[cur_pos[i]] = node["rep_frames"][i] 
    
    # rep_pos = list(set(cur_pos))
    # rep_pos.sort()
    # rep_frames = []
    # for pos in rep_pos:
    #     rep_frames.append(clear_dict[pos])
    # node['rep_pos'] = rep_pos
    # node['rep_frames'] = rep_frames

    all_lines[node['name']] = ac.alignments[0].copy()
    all_adj_dists[node['name']] = deepcopy(adj_dist)