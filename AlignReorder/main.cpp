//example.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace py = pybind11;
using std::cout;

const int min_non_valid_gap = 2;
const double cluster_inner_dist_factor = 0.6;

inline double sqr(double x) {
    return x * x;
}

template <typename T>
inline T min(const T&a, const T& b) {
    return a < b ? a : b;
}

template <typename T>
inline T max(const T&a, const T& b) {
    return a > b ? a : b;
}

template <typename T>
std::vector<int> argsort(const std::vector<T> dist) {
    std::vector<std::pair<T, int>> vec;
    int n = dist.size();
    for (auto i = 0; i < n; ++i) {
        vec.push_back(std::make_pair(dist[i], i));
    }
    std::sort(vec.begin(), vec.end());
    std::vector<int> ret;
    for (auto i = 0; i < n; ++i) {
        ret.push_back(vec[i].second);
    }
    return ret;
}


template <typename T>
std::vector<T> getCol(const std::vector<std::vector<T>> &vec, int j) {
    std::vector<T> ret;
    int n = vec.size();
    for (auto i = 0; i < n; ++i) {
        ret.push_back(vec[i][j]);
    }
    return ret;
}

class Node {
public:
    int x;
    std::vector<int> s;

    Node(){}
    Node(int _x, const std::vector<int>& _s): x(_x), s(_s) {}
    bool operator==(const Node& p) const {
        if (x != p.x || s.size() != p.s.size()) return false;
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] != p.s[i]) return false;
        }
        return true;
    }
};

struct node_hash {
    size_t operator()(const Node& p) const{
        size_t ret = p.x + 1;
        int m = p.s.size();
        for (int i = 0; i < p.s.size(); ++i) {
            ret = ret * m + p.s[i];
        }
        return ret;
    }
};

double count_reverse_pair(
    const std::vector<int>& state1,
    const std::vector<int>& state2,
    std::vector<int>* v,
    std::vector<double> *d = nullptr) {

    int n = state1.size();
    std::vector<int> p, q;
    for (int i = 0; i < n; ++i) {
        p.push_back(0);
        q.push_back(0);
    }

    for (int i = 0; i < n; ++i) {
        p[state1[i]] = i;
        q[state2[i]] = i;
    }

    double cnt = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double cost = 1;
            if (!v->at(i) || !v->at(j)) {
                cost = 0.5;
            }
            if ((p[i] < p[j]) != (q[i] < q[j])) {
                if (!d) {
                    cnt += cost;
                } else {
                    cnt += abs(d->at(i) - d->at(j)) * cost;
                }
            }
        }
    }
    return cnt;
}

std::vector<std::vector<int>> AlignOrdering(
    std::vector<std::vector<double>> input_dist,
    std::vector<std::vector<int>> vis_matrix,
    double alpha) {
    int n_cols = input_dist[0].size();
    int n_rows = input_dist.size();
    std::vector<int> order;
    for (int i = 0; i < n_rows; ++i) {
        order.push_back(i);
    }
    std::vector<std::vector<int>> all_states;
    do {
        all_states.push_back(order);
    } while (next_permutation(order.begin(), order.end()));

    std::unordered_map<Node, Node, node_hash> prev;
    std::unordered_map<Node, double, node_hash> dist;
    std::vector<std::vector<int>> col_states;
    for (int j = 0; j < n_cols; ++j) {
        col_states.push_back(argsort(getCol(input_dist, j)));
        for (int i = 0; i < n_rows; ++i) {
            input_dist[i][j] *= (n_rows - 1);
        }
    }

    std::vector<std::vector<int>> mat_col;
    std::vector<std::vector<double>> dist_col;
    for (int j = 0; j < n_cols; ++j) {
        mat_col.push_back(getCol(vis_matrix, j));
        dist_col.push_back(getCol(input_dist, j));
    }

    std::vector<Node> Q;
    int head = 0;
    // 0 => history
    auto start = Node(-1, all_states[0]);
    // auto start = Node(-1, history_state);
    dist[start] = 0;
    Q.push_back(start);

    double min_dist = 0;
    Node end_state;
    auto prev_state = start.s;
    for (int i = 0; i < n_cols; ++i) {
        double min_cost = 1e10;
        std::vector<int> min_state;
        for (auto state: all_states) {
            double self_cost = count_reverse_pair(state, col_states[i], &mat_col[i], &dist_col[i]);
            double between_cost = 0;
            if (i > 0) {
                between_cost = count_reverse_pair(state, prev_state, &mat_col[i - 1]);
            }
            double cost = between_cost + self_cost * alpha;
            if (cost < min_cost) {
                min_cost = cost;
                min_state = state;
            }
        }
        auto state = min_state;
        double cost = min_cost;
        Node next_state = Node(i, state);
        min_dist += cost;
        dist[next_state] = min_dist;
        prev[next_state] = Node(i - 1, prev_state);
        prev_state = state;
        if (i == n_cols - 1) {    
            end_state = next_state;
        }
    }

    std::unordered_set<Node, node_hash> inqueue;
    inqueue.insert(start);

    while (head < Q.size()) {
        auto curr_state = Q[head];
        int col = curr_state.x;
        auto state1 = curr_state.s;
        head += 1;
        double d = dist[curr_state];
        inqueue.erase(curr_state);
        if (d > min_dist) {
            continue;
        }
        for (auto state2: all_states) {
            double between_cost = 0;
            if (col > -1) {
                between_cost = count_reverse_pair(state1, state2, &mat_col[col]);
            }
            double self_cost = count_reverse_pair(state2, col_states[col + 1], &mat_col[col + 1], &dist_col[col + 1]);
            double cost = between_cost + self_cost * alpha;
            auto next_state = Node(col + 1, state2);
            if (col + 1 == n_cols - 1) {
                if (d + cost < min_dist) {
                    min_dist = d + cost;
                    dist[next_state] = min_dist;
                    prev[next_state] = curr_state;
                    end_state = next_state;
                }
            } else {
                if (d + cost > min_dist) {
                    continue;
                }
                if (dist.find(next_state) == dist.end()) {
                    dist[next_state] = d + cost;
                    prev[next_state] = curr_state;
                    Q.push_back(next_state);
                    inqueue.insert(next_state);
                } else if (d + cost < dist[next_state]) {
                    dist[next_state] = d + cost;
                    prev[next_state] = curr_state;
                    if (inqueue.find(next_state) == inqueue.end()) {
                        Q.push_back(next_state);
                        inqueue.insert(next_state);
                    }
                }
            }
        }
    }
    auto state = end_state;
    std::vector<std::vector<int>> ret;
    while (1) {
        ret.push_back(state.s);
        if (prev.find(state) == prev.end()) {
            break;
        }
        state = prev[state];
    }
    ret.pop_back();
    std::reverse(ret.begin(), ret.end());
    return ret;
}

PYBIND11_MODULE(AlignReorder, m) {
    m.doc() = "Align Ordering algorithm"; // optional module docstring
    m.def("AlignOrdering", &AlignOrdering, "AlignOrdering function");
}