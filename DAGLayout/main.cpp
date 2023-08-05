//example.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <utility>

namespace py = pybind11;
using std::cout;

struct buffer_info {
    void *ptr;
    ssize_t itemsize;
    std::string format;
    ssize_t ndim;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
}; 

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

std::vector<std::vector<double>> DAGLayout(
    const std::vector<std::vector<double>> &_dist,
    const std::vector<std::vector<int>> &_valid,
    const std::vector<int> &labels,
    int grid_n, double alpha, double beta) {
    int n_row = _dist.size();
    int n_col = _dist[0].size();

    std::vector<std::vector<double>> dist;
    for (int j = 0; j < n_col; ++j) {
        std::vector<double> row;
        for (int i = 0; i < n_row; ++i) {
            row.push_back(_dist[i][j]);
        }
        dist.push_back(row);
    }

    std::vector<std::vector<int>> valid;
    for (int j = 0; j < n_col; ++j) {
        std::vector<int> row;
        for (int i = 0; i < n_row; ++i) {
            row.push_back(_valid[i][j]);
        }
        valid.push_back(row);
    }

    for (int j = 0; j < n_row; ++j) {
        int left = 0, right = n_col - 1;
        while (left <= right && valid[left][j] == -1) ++left;
        while (left <= right && valid[right][j] == -1) --right;
        if (left > 0) --left;
        if (right < n_col - 1) ++right;
        for (int i = left; i <= right; ++i) {
            valid[i][j] += 1;
        }
    }
    /*
    for (int i = 0; i < n_col; ++i) {
        int last_valid = -1;
        for (int j = n_row - 1; j >= 0; --j) {
            if (valid[i][j] > 0) {
                last_valid = j;
            } else if (last_valid != -1) {
                dist[i][last_valid] += dist[i][j] / 2;
                dist[i][j] = dist[i][j] / 2;
            }
        }
        last_valid = -1;
        for (int j = 0; j < n_row; ++j) {
            if (valid[i][j] > 0) {
                last_valid = j;
            } else if (last_valid != -1) {
                double t = dist[i][j];
                dist[i][last_valid] += t;
                dist[i][j] = 0;
            }
        }
    }
    */

    double grid_step = 1.0 / grid_n;
    int most_valid_column = -1, max_cnt = -1;
    for (int i = 0; i < n_col; ++i) {
        int cnt = 0;
        for (int j = 0; j < n_row; ++j) {
            cnt += valid[i][j];
        }
        if (cnt > max_cnt) {
            max_cnt = cnt;
            most_valid_column = i;
        }
    }

    for (int left = 0; left < n_row; ++left) {
        if (left == 0 || labels[left] != labels[left - 1]) {
            int right = left + 1;
            while (right < n_row && labels[right] == labels[left]) ++right;
            for (int i = 0; i < n_col; ++i) {
                double totd = 0;
                for (int j = left; j < right; ++j) {
                    totd += dist[i][j];
                }
                if (totd > 0) {
                    double remain = totd * (1 - cluster_inner_dist_factor);
                    for (int j = left; j < right; ++j) {
                        dist[i][j] *= cluster_inner_dist_factor;
                    }
                    dist[i][left] += remain * 0.5;
                    if (right < n_row) {
                        dist[i][right] += remain * 0.5;
                    }
                }
            }
        }
    }

    double max_totd = 1;
    for (int i = 0; i < n_col; ++i) {
        double totd = 0;
        for (int j = 0; j < n_row; ++j) {
            totd += dist[i][j];
        }
        if (totd > max_totd) {
            max_totd = totd;
        }
        // cout << i << ' ' << totd << std::endl;
    }

    for (int i = 0; i < n_col; ++i) {
        for (int j = 0; j < n_row; ++j) {
            dist[i][j] /= max_totd;
        }
    }

    std::vector<std::pair<double, std::pair<int, int>>> all_dist;
    for (int i = 0; i < n_col; ++i) {
        for (int j = 0; j < n_row; ++j) {
            if (dist[i][j] > 0) {
                all_dist.push_back(std::make_pair(dist[i][j], std::make_pair(i, j)));
            }
        }
    }

    std::vector<std::vector<double>> dist_quantile;
    for (int i = 0; i < n_col; ++i) {
        std::vector<double> row;
        for (int j = 0; j < n_row; ++j) {
            row.push_back(1);
        }
        dist_quantile.push_back(row);
    }
    for (int i = 0, k = 0; i < all_dist.size(); ++i) {
        auto p = all_dist[i].second;
        if (i > 0 && all_dist[i].first != all_dist[i - 1].first) {
            k = i;
        }
        dist_quantile[p.first][p.second] = 1 - k / (all_dist.size() + 1);
    }

    for (int i = most_valid_column + 1; i < n_col; ++i) {
        for (int j = 0; j < n_row; ++j) {
            if (dist[i - 1][j] > 0.1 && dist[i - 1][j] < dist[i][j]) {
                dist[i - 1][j] = dist[i][j];
            }
        }
    }
    for (int i = most_valid_column - 1; i >= 0; --i) {
        for (int j = 0; j < n_row; ++j) {
            if (dist[i + 1][j] > 0.1 && dist[i + 1][j] < dist[i][j]) {
                dist[i + 1][j] = dist[i][j];
            }
        }
    }
    // std::cerr << "33333" << std::endl;

    std::vector<std::vector<double>> y;
    for (int i = 0; i < n_col; ++i) {
        std::vector<double> row;
        for (int j = 0; j < n_row; ++j) {
            row.push_back(0);
        }
        y.push_back(row);
    }
    double *f;
    f = new double[n_row * grid_n];
    int *pre;
    pre = new int[n_row * grid_n];
    // std::cout << "444444" << std::endl;

    auto optimize = [&](int curr, int prev) {

        for (int j = 0; j < n_row; ++j) {
            for (int k = 0; k < grid_n; ++k) {
                f[j * grid_n + k] = 1e10;
                pre[j * grid_n + k] = -1;
            }
        }

        for (int k = 0; k < grid_n; ++k) {
            double t = k * grid_step;
            if (prev != -1) {
                f[k] = sqr(t - y[prev][0]);
            } else {
                f[k] = 0;
            }
        }

        for (int j = 0; j < n_row; ++j) {
            for (int k = 0; k < grid_n - j * 2; ++k) {
                double t = k * grid_step;
                double ft = 0;
                int st, ed;
                if (prev != -1) {
                    if (t == y[prev][j]) {
                        ft = -beta * sqr(grid_step);
                    } else if (valid[prev][j] >= 0) {
                        ft = sqr(t - y[prev][j]) * alpha;
                    }
                } else {
                    ft = 0;
                }
                double delta = dist[curr][j];
                //if (valid[curr][j] >= 0) {
                    st = min(grid_n, k + 2);
                    ed = grid_n;
                //} else {
                //    st = min(grid_n, k + min_non_valid_gap);
                //    ed = st + 1;
                //}
                for (int l = st; l < ed; ++l) {
                    double d = (l - k) * grid_step;
                    double cost = f[(j - 1) * grid_n + l] + ft + sqr(d - delta) * sqr(dist_quantile[curr][j]);
                    if (cost < f[j * grid_n + k]) {
                        f[j * grid_n + k] = cost;
                        pre[j * grid_n + k] = l;
                    }
                }
            }
        }

        int k = 0;
        for (int j = 0; j < grid_n - 1; ++j) {
            if (f[(n_row - 1) * grid_n + j] < f[(n_row - 1) * grid_n + k]) {
                k = j;
            }
        }

        for (int j = n_row - 1; j >= 0; --j) {
            y[curr][j] = k * grid_step;
            k = pre[j * grid_n + k];
        }
    };
    // std::cout << "5555" << std::endl;
    
    optimize(most_valid_column, -1);

    for (int i = most_valid_column + 1; i < n_col; ++i) {
        optimize(i, i - 1);
    }
    // std::cout << "6666" << std::endl;

    for (int i = most_valid_column - 1; i >= 0; --i) {
        optimize(i, i + 1);
    }
    // std::cout << "77777" << std::endl;

    for (int i = 1; i < n_col - 1; ++i) {
        for (int j = 0; j < n_row; ++j) {
            if (y[i][j] > y[i - 1][j] && y[i][j] > y[i + 1][j]) {
                y[i][j] = max(y[i - 1][j], y[i + 1][j]);
            }
            if (y[i][j] < y[i - 1][j] && y[i][j] < y[i + 1][j]) {
                y[i][j] = min(y[i - 1][j], y[i + 1][j]);
            }
        }
    }

    /*
    for (int j = 0; j < n_row; ++j) {
        for (int i = 0; i < n_col - 1; ++i) if (valid[i][j] > 0 && valid[i + 1][j] == 0) {
            for (int k = i + 1; k < n_col - 1; ++k) if (valid[k][j] == 0 && valid[k + 1][j] > 0) {
                int st = i, ed = k + 1;
                if (y[st][j] != y[ed][j]) {
                    while (y[st][j] == y[st + 1][j]) ++st;
                    while (y[ed][j] == y[ed - 1][j]) --ed;
                    double mid_y = (y[st][j] + y[ed][j]) / 2;
                    int mid = (st + ed) / 2;
                    while (st + 1 < mid){ y[st + 1][j] = y[st][j]; ++st; }
                    while (ed - 1 > mid){ y[ed - 1][j] = y[ed][j]; --ed; }
                    y[mid][j] = mid_y;
                }
            }
        }
    }
    */
    std::vector<std::vector<double>> ret;
    for (int j = 0; j < n_row; ++j) {
        std::vector<double> row;
        for (int i = 0; i < n_col; ++i) {
            row.push_back(y[i][j]);
        }
        ret.push_back(row);
    }

    delete pre;
    delete f;
    return ret;
    // y *= max_sum
}

PYBIND11_MODULE(DAGLayout, m) {
    m.doc() = "DAGLayout algorithm"; // optional module docstring
    m.def("DAGLayout", &DAGLayout, "A function");
}