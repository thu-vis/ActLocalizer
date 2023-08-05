from application.views.model_utils import VideoModel
from application.views.utils.config_utils import config
import time
import numpy as np



model = VideoModel(config.thumos19)
model.prop_model.load_prerun_result()
model.cluster_helper.call_test()


exit(0)
time1 = time.time()

class_label = 0
action_id = "686-10"
n_neighbors = 5

action_ids = model.cluster_helper.get_actions(class_label)
print(action_ids[0])
print(len(action_ids))

dist = model.cluster_helper.get_alignment_of_anchor_action(class_label, action_id, n_neighbors)
dist = np.array(dist)

from itertools import permutations
# both p1 and p2 should be a permutation of 0-n sequence.
def count_reverse_pair(p, q, d = None):
    n = len(p)
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (p[i] < p[j]) != (q[i] < q[j]):
                if d is None:
                    cnt += 1
                else:
                    cnt += abs(d[i] - d[j])
    return cnt

# alpha is a parameter to balance the between cost and self cost.
def get_order_by_dist_matrix(input_dist, alpha = 1.0):
    input_dist = np.array(input_dist)
    n_cols = input_dist.shape[1]
    n_rows = input_dist.shape[0]
    order = [i for i in range(n_rows)]
    all_states = [x for x in permutations(order)]
    col_states = []
    for i in range(n_cols):
        col_states.append(tuple(np.argsort(input_dist[:, i]).tolist()))
        min_value = input_dist[:, i].min()
        max_value = input_dist[:, i].max()
        input_dist[:, i] = (input_dist[:, i] - min_value) / (max_value - min_value) * (n_rows - 1)
    print(col_states)
    dist = {}
    prev = {}

    Q = []
    head = 0
    start = (0, col_states[0])
    dist[start] = 0
    prev[start] = None
    Q.append(start)

    min_dist = 1e10
    end_state = None
    while head < len(Q):
        curr_state = Q[head]
        col, state1 = curr_state
        head += 1
        d = dist[curr_state]
        if d > min_dist:
            continue
        for state2 in all_states:
            between_cost = count_reverse_pair(state1, state2)
            self_cost = count_reverse_pair(state2, col_states[col + 1], input_dist[:, col + 1])
            cost = between_cost * between_cost + self_cost * self_cost * alpha
            next_state = (col + 1, state2)
            if col + 1 == n_cols - 1:
                if d + cost < min_dist:
                    min_dist = d + cost
                    prev[next_state] = curr_state
                    end_state = next_state
            else:
                if next_state not in dist:
                    dist[next_state] = d + cost
                    prev[next_state] = curr_state
                    Q.append(next_state)
                elif d + cost < dist[next_state]:
                    dist[next_state] = d + cost
                    prev[next_state] = curr_state
    state = end_state
    ret = []
    while state != None:
        ret.append(state[1])
        state = prev[state]
    ret = ret[::-1]
    return ret

print(get_order_by_dist_matrix(dist))

time2 = time.time()
print("Using time {} s".format(time2 - time1))

