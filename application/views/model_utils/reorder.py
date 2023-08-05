import numpy as np
import math
from itertools import permutations
try:
    from Reorder import reorder
    from AlignReorder import AlignReorder
    from AlignReorder_o import AlignReorder as AlignReorder_o
except:
    None
from IPython import embed

digits_length = 4
n_quantify = 1
adj_boundary = 2
# adj_boundary = 1

from matplotlib import pyplot as plt
def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    plt.hist(myList,100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.savefig("1.jpg")

def quantify_dists(dists, n):
    return np.minimum(np.ceil(np.floor(dists * 2 * n) / 2), n) / n

# 1. distance and reorder for adjust mode
def get_dists_matrix(lines, aligns, afmatrix, alpha=0.9, case=-1):
    """
    params:
        lines - alignments to do reordering (n,)
        aligns - align results (n,width)
        afmatrix - afmatrix for calculate S and L
        alpha - similarity args
    return:
        matrix - [i, j, width]
    """
    A_list = []

    # 1. init frame to index list for lines
    n = len(lines)
    width = len(aligns[0])
    fti_lists = []
    for align in aligns:
        fti_list = []
        for i in range(width):
            if align[i] != -1:
                fti_list.append(i)
        fti_lists.append(fti_list)

    # 2. calculate similarity and distance
    def A(a, b, single_line = False):    # judge afmatrix
        if b in afmatrix[a] and \
            (afmatrix[a][b] >= adj_boundary or single_line and afmatrix[a][b] >= 1):
            # if case > 0:
            #     A_list.append(afmatrix[a][b])
            return afmatrix[a][b]
        return 0

    def get_frame_dists(i,j):
        # 0. init and get target align
        align1, align2 = aligns[i], aligns[j]
        line1, line2 = lines[i], lines[j]
        fti1, fti2 = fti_lists[i], fti_lists[j]
        dists = []

        # 1. get c list for calulate
        c1 = np.full((line1.width,), np.inf)
        c2 = np.full((line2.width,), np.inf)
        single_line = False
        if line1.size + line2.size == 2:
            single_line = True
        for a in range(line1.width):
            for b in range(line2.width):
                if A(line1.index + a, line2.index + b, single_line):
                    c = abs(fti1[a] - fti2[b])
                    if c < c1[a]:
                        c1[a] = c
                    if c < c2[b]:
                        c2[b] = c
        
        # 2. calculate dists of frames
        def fc(a, b):    
            # get c and calculate 1/2 * (1/(1+c1^2) + 1/(1+c2^2))
            ret = 0
            if c1[a] != np.inf:
                ret += 1 / (1 + pow(c1[a],2) / 10)
            if c2[b] != np.inf:
                ret += 1 / (1 + pow(c2[b],2) / 10)
            return ret / 2

        def s(a, b):    # similarity of frame
            # if A(line1.index + a, line2.index + b):
            #     return 1
            # return 0
            return (alpha * fc(a, b) + 1- alpha) \
                * 1 # (line1.features[a] @ line2.features[b])

        a, b = -1, -1
        for w in range(width):
            if align1[w] == -1 and align2[w] == -1:
                dists.append(-1)
                continue
            if align1[w] == -1:
                b += 1
                dists.append(-1)
            elif align2[w] == -1:
                a += 1
                dists.append(-1)
            else:
                a += 1
                b += 1
                dists.append(1 - round(s(a, b), digits_length))

        # 3. do interpolation for null pos
        left = np.full((width,), -1)
        right = np.full((width,), -1)
        for w in range(width):
            if dists[w] != -1:
                left[w] = w
            elif w > 0:
                left[w] = left[w-1]
            _w = width - 1 - w
            if dists[_w] != -1:
                right[_w] = _w
            elif _w < width - 1:
                right[_w] = right[_w+1]

        for w in range(width):
            if dists[w] == -1:
                value = 0
                bound = 0
                if left[w] != -1 and right[w] != -1:
                    value += (w - left[w]) * dists[right[w]]
                    value += (right[w] - w) * dists[left[w]]
                    bound += (right[w] - left[w])
                else:
                    if left[w] != -1:
                        value += dists[left[w]]
                    if right[w] != -1:
                        value += dists[right[w]]
                    bound = 1
                if bound != 0:
                    dists[w] = round(value / bound, digits_length)
                else:
                    dists[w] = 1
        return dists
    
    ret = np.zeros((n,n,width))
    for i in range(n):
        for j in range(i):
            dists = get_frame_dists(i, j)
            ret[i, j] = np.array(dists)
            ret[j, i] = ret[i, j]
    
    # if case > 0:
    #     draw_hist(A_list, 'Align Times', 'align number', 'times', 0, 8, 0, 50)
    #     A_list = []
    
    ret = quantify_dists(ret, n_quantify)
    return ret

def get_order_and_dists_adj(lines, aligns, afmatrix, case=-1):
    frame_dists = get_dists_matrix(lines, aligns, afmatrix, case=case)
    dists = frame_dists.sum(axis=2)
    order = reorder.reorder_func(dists).astype(np.int32).tolist()
    if case == 10:
        _order = [3, 0, 4, 2, 1]
        order = list(map(lambda x: order[x], _order))
    ln, width = frame_dists.shape[0], frame_dists.shape[2]
    adj_dists = np.zeros((ln-1, width))
    for i in range(ln-1):
        adj_dists[i] = frame_dists[order[i+1], order[i]]
    return order, adj_dists

def get_adjancent_dists(lines, aligns, afmatrix, alpha=0.9):
    """
    params:
        lines - alignments to do reordering (n,)
        aligns - align results (n,width)
        afmatrix - afmatrix for calculate S and L
        alpha - similarity args
    return:
        matrix - [i-1, width]
    """
    # 1. init frame to index list for lines
    n = len(lines)
    width = len(aligns[0])
    fti_lists = []
    for align in aligns:
        fti_list = []
        for i in range(width):
            if align[i] != -1:
                fti_list.append(i)
        fti_lists.append(fti_list)

    # 2. calculate similarity and distance
    def A(a, b, single_line = False):    # judge afmatrix
        if b in afmatrix[a] and \
            (afmatrix[a][b] >= adj_boundary or single_line and afmatrix[a][b] >= 1):
            return afmatrix[a][b]
        return 0

    def get_frame_dists(i,j):
        # 0. init and get target align
        align1, align2 = aligns[i], aligns[j]
        line1, line2 = lines[i], lines[j]
        fti1, fti2 = fti_lists[i], fti_lists[j]
        dists = []

        # # 1. get c list for calulate
        c1 = np.full((line1.width,), np.inf)
        c2 = np.full((line2.width,), np.inf)
        single_line = False
        if line1.size + line2.size == 2:
            single_line = True
        for a in range(line1.width):
            for b in range(line2.width):
                if A(line1.index + a, line2.index + b, single_line):
                    c = abs(fti1[a] - fti2[b])
                    if c < c1[a]:
                        c1[a] = c
                    if c < c2[b]:
                        c2[b] = c
        
        # 2. calculate dists of frames
        def fc(a, b):    
            # get c and calculate 1/2 * (1/(1+c1^2) + 1/(1+c2^2))
            ret = 0
            if c1[a] != np.inf:
                ret += 1 / (1 + pow(c1[a],2) / 10)
            if c2[b] != np.inf:
                ret += 1 / (1 + pow(c2[b],2) / 10)
            return ret / 2

        def s(a, b):    # similarity of frame
            # if A(line1.index + a, line2.index + b):
            #     return 1
            # return 0
            return (alpha * fc(a, b) + 1- alpha) \
                * 1 # (line1.features[a] @ line2.features[b])

        a, b = -1, -1
        for w in range(width):
            if align1[w] == -1 and align2[w] == -1:
                dists.append(-1)
                continue
            if align1[w] == -1:
                b += 1
                dists.append(-1)
            elif align2[w] == -1:
                a += 1
                dists.append(-1)
            else:
                a += 1
                b += 1
                dists.append(1 - round(s(a, b), digits_length))

        # 3. do interpolation for null pos
        left = np.full((width,), -1)
        right = np.full((width,), -1)
        for w in range(width):
            if dists[w] != -1:
                left[w] = w
            elif w > 0:
                left[w] = left[w-1]
            _w = width - 1 - w
            if dists[_w] != -1:
                right[_w] = _w
            elif _w < width - 1:
                right[_w] = right[_w+1]

        for w in range(width):
            if dists[w] == -1:
                value = 0
                bound = 0
                if left[w] != -1 and right[w] != -1:
                    value += (w - left[w]) * dists[right[w]]
                    value += (right[w] - w) * dists[left[w]]
                    bound += (right[w] - left[w])
                else:
                    if left[w] != -1:
                        value += dists[left[w]]
                    if right[w] != -1:
                        value += dists[right[w]]
                    bound = 1
                if bound != 0:
                    dists[w] = round(value / bound, digits_length)
                else:
                    dists[w] = 1
        return dists
    
    adj_dists = np.zeros((n-1, width))
    for i in range(n-1):
        adj_dists[i] = np.array(get_frame_dists(i+1, i))
    adj_dists = quantify_dists(adj_dists, n_quantify)
    return adj_dists

def get_detail_order(lines, aligns, afmatrix):
    frame_dists = get_dists_matrix(lines, aligns, afmatrix)
    # 1. get orders
    tsp_orders = []
    avg_order = list(range(frame_dists.shape[0]))
    for i in range(frame_dists.shape[2]):
        dists = frame_dists[:,:,i]
        order = reorder.reorder_func(dists).astype(np.int32)
        order = order.tolist()
        _order = order[::-1]
        if count_reverse_pair(order, avg_order) < count_reverse_pair(_order, avg_order):
            tsp_orders.append(order)
        else:
            tsp_orders.append(_order)

    input_dist = np.array(tsp_orders).T.astype(np.float32)
    vis_matrix = np.zeros(input_dist.shape)
    for i in range(input_dist.shape[0]):
        vec = np.nonzero(np.array(aligns[i]) + 1)[0]
        lf, rt = vec.min(), vec.max() + 1
        vis_matrix[i][lf: rt] = 1
    vis_matrix = vis_matrix.astype(np.int32)
    orders = AlignReorder_o.AlignOrdering(input_dist, vis_matrix, .75)

    # 1.5 judge cross and orders
    cols = len(orders)
    sum_cross = 0
    for i in range(cols):
        if i == 0:
            continue
        sum_cross += count_reverse_pair_vis(orders[i - 1], orders[i], vis_matrix[:, i-1], vis_matrix[:, i])
    if sum_cross == 0:
        orders = [list(range(len(order))) for order in orders]

    # 2. get adj dists
    adj_dists = []
    for i in range(frame_dists.shape[2]):
        order = orders[i]
        adj_dist = []
        for j in range(frame_dists.shape[0]):
            if j == 0:
                continue
            adj_dist.append(frame_dists[order[j]][order[j - 1]][i])
        adj_dists.append(adj_dist)
    adj_dists = np.array(adj_dists).T.tolist()
    return orders, adj_dists

# 2. distance and reorder for center mode 
def get_center_dists(lines, aligns, afmatrix, alpha=0.9):
    width = len(aligns[0])
    n = len(lines)

    ret_dist = []

    # # get frame to align index list
    fti_lists = []
    for align in aligns:
        fti_list = []
        for i in range(width):
            if align[i] != -1:
                fti_list.append(i)
        fti_lists.append(fti_list)

    def A(a, b):    # judge afmatrix
        if b in afmatrix[a]:
            return afmatrix[a][b]
        return 0

    def get_frame_dists(i,j):
        # 0. init and get target align
        align1, align2 = aligns[i], aligns[j]
        line1, line2 = lines[i], lines[j]
        fti1, fti2 = fti_lists[i], fti_lists[j]
        dists = []

        # 1. get c list for calulate
        c1 = np.full((line1.width,), np.inf)
        c2 = np.full((line2.width,), np.inf)
        for a in range(line1.width):
            for b in range(line2.width):
                if A(line1.index + a, line2.index + b):
                    c = abs(fti1[a] - fti2[b])
                    if c < c1[a]:
                        c1[a] = c
                    if c < c2[b]:
                        c2[b] = c
        
        # 2. calculate dists of frames
        def fc(a, b):    
            # get c and calculate 1/2 * (1/(1+c1^2) + 1/(1+c2^2))
            ret = 0
            if c1[a] != np.inf:
                ret += 1 / (1 + pow(c1[a],2) / 10)
            if c2[b] != np.inf:
                ret += 1 / (1 + pow(c2[b],2) / 10)
            return ret / 2

        def s(a, b):    # similarity of frame
            # if A(line1.index + a, line2.index + b):
            #     return 1
            # return 0
            return (alpha * fc(a, b) + 1- alpha) \
                * 1 # (line1.features[a] @ line2.features[b])

        a, b = -1, -1
        for w in range(width):
            if align1[w] == -1 and align2[w] == -1:
                dists.append(-1)
                continue
            if align1[w] == -1:
                b += 1
                dists.append(-1)
            elif align2[w] == -1:
                a += 1
                dists.append(-1)
            else:
                a += 1
                b += 1
                dists.append(1 - round(s(a, b), digits_length))

        # 3. do interpolation for null pos
        left = np.full((width,), -1)
        right = np.full((width,), -1)
        for w in range(width):
            if dists[w] != -1:
                left[w] = w
            elif w > 0:
                left[w] = left[w-1]
            _w = width - 1 - w
            if dists[_w] != -1:
                right[_w] = _w
            elif _w < width - 1:
                right[_w] = right[_w+1]

        for w in range(width):
            if dists[w] == -1:
                value = 0
                bound = 0
                if left[w] != -1 and right[w] != -1:
                    value += (w - left[w]) * dists[right[w]]
                    value += (right[w] - w) * dists[left[w]]
                    bound += (right[w] - left[w])
                else:
                    if left[w] != -1:
                        value += dists[left[w]]
                    if right[w] != -1:
                        value += dists[right[w]]
                    bound = 1
                if bound != 0:
                    dists[w] = round(value / bound, digits_length)
                else:
                    dists[w] = 1
        return dists
    
    for i in range(1, n):
        ret_dist.append(get_frame_dists(0, i))

    ret = quantify_dists(np.array(ret_dist), 1)
    ret_dist = ret.tolist()
    return ret_dist

# both p1 and p2 should be a permutation of 0-n sequence.
def count_reverse_pair(state1, state2):
    n = len(state1)
    p, q = list(range(n)), list(range(n))
    for pos in range(n):
        p[state1[pos]] = pos
        q[state2[pos]] = pos

    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            cost = 1
            if (p[i] < p[j]) != (q[i] < q[j]):
                cnt += cost
    return cnt

def count_reverse_pair_vis(state1, state2, v1, v2):
    n = len(state1)
    p, q = list(range(n)), list(range(n))
    for pos in range(n):
        p[state1[pos]] = pos
        q[state2[pos]] = pos

    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            cost = 1
            if not v1[i] or not v1[j] or not v2[i] or not v2[j] :
                cost = 0
            if (p[i] < p[j]) != (q[i] < q[j]):
                cnt += cost
    return cnt

def count_reverse_pairs(state1, state2, v = None, d = None):
    n = len(state1)
    p, q = list(range(n)), list(range(n))
    for pos in range(n):
        p[state1[pos]] = pos
        q[state2[pos]] = pos

    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            cost = 1
            if not v[i] or not v[j]:
                cost = 0.5
            if (p[i] < p[j]) != (q[i] < q[j]):
                if d is None:
                    cnt += cost
                else:
                    cnt += abs(d[i] - d[j]) * cost
    return cnt

# alpha is a parameter to balance the between cost and self cost.
def get_order_by_dist_matrix_ctr(input_dist, vis_matrix, alpha = 2):
    # TODO: reorder cost should Change from square term to primary term
    # ps: alpha = 2 means eta = 0.5 in paper
    dist = np.array(input_dist).astype(np.float32)
    mat = np.array(vis_matrix).astype(np.int32)
    return AlignReorder.AlignOrdering(dist, mat, alpha)
    input_dist = np.array(input_dist)
    vis_matrix = np.array(vis_matrix)
    n_cols = input_dist.shape[1]
    n_rows = input_dist.shape[0]
    order = [i for i in range(n_rows)]
    all_states = [x for x in permutations(order)]
    col_states = []
    for i in range(n_cols):
        col_states.append(tuple(np.argsort(input_dist[:, i]).tolist()))
        input_dist[:, i] *= (n_rows - 1)
    # print("col_state:", col_states)
    dist = {}
    prev = {}

    Q = []
    head = 0
    start = (-1, tuple(range(n_rows)))
    dist[start] = 0
    prev[start] = None
    Q.append(start)

    min_dist = 0
    end_state = None
    prev_state = start[1]
    for i in range(n_cols):
        min_cost = 1e10
        min_state = -1
        for state in all_states:
            self_cost = count_reverse_pairs(state, col_states[i], vis_matrix[:, i], input_dist[:, i])
            between_cost = count_reverse_pairs(state, prev_state, vis_matrix[:, i - 1]) if i > 0 else 0
            cost = between_cost + self_cost * alpha
            if cost < min_cost:
                min_cost = cost
                min_state = state
        state = min_state
        cost = min_cost
        next_state = (i, state)
        min_dist += cost
        dist[next_state] = min_dist
        prev[next_state] = (i - 1, prev_state)
        prev_state = state
    end_state = next_state
    inqueue = {}
    inqueue[start] = True

    while head < len(Q):
        curr_state = Q[head]
        col, state1 = curr_state
        head += 1
        d = dist[curr_state]
        inqueue[curr_state] = False
        if d > min_dist:
            continue
        for state2 in all_states:
            between_cost = count_reverse_pairs(state1, state2, vis_matrix[:, col]) if col > -1 else 0
            self_cost = count_reverse_pairs(state2, col_states[col + 1], vis_matrix[:, col + 1], input_dist[:, col + 1])
            cost = between_cost + self_cost * alpha
            next_state = (col + 1, state2)
            if col + 1 == n_cols - 1:
                if d + cost < min_dist:
                    min_dist = d + cost
                    dist[next_state] = min_dist
                    prev[next_state] = curr_state
                    end_state = next_state
            else:
                if d + cost > min_dist:
                    continue
                if next_state not in dist:
                    dist[next_state] = d + cost
                    prev[next_state] = curr_state
                    Q.append(next_state)
                    inqueue[next_state] = True
                elif d + cost < dist[next_state]:
                    dist[next_state] = d + cost
                    prev[next_state] = curr_state
                    if not inqueue.get(next_state, False):
                        Q.append(next_state)
                        inqueue[next_state] = True
    state = end_state
    ret = []
    dists = []
    while state != None:
        dists.append(dist[state])
        ret.append(state[1])
        state = prev[state]
    ret = ret[::-1]
    dists = dists[::-1]
    return ret[1:]


if __name__ == "__main__":
    '''
    dists = np.array([[0. , 0. , 0.5, 1. , 1. , 0.5, 0. ],
                      [1. , 1. , 0.5, 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 1. , 0.5, 0. ],
                      [1. , 1. , 1. , 1. , 1. , 1. , 1. ]])
    '''
    #import pickle
    #dists = pickle.load(open('/home/jiashu/VideoVis/test/test.pkl', 'rb')
    dists = [
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0]
    ]

    ret = get_order_by_dist_matrix_ctr(dists, alpha=1)
    print(ret)