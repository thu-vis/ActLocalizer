//example.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <utility>
#include <assert.h>

//#define NDEBUG  

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

const int min_valid_gap = 6;

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
    int grid_n, double alpha, double beta,
    double between_cluster_margin = 0.2,
    double inner_cluster_alpha = 0.8) {
    int n_row = _dist.size();
    int n_col = _dist[0].size();

    #define DIST(a, b, c) dist[(a) * n_row * n_row + (max(b, 0)) * n_row + (max(c, 0))]
    #define VALID(a, b) valid[(a) * n_row + (b)]
    #define V3I(a, b, c) ((a) * n_row * n_row + (max(b, 0)) * n_row + (max(c, 0)))
    #define V2I(a, b) ((a) * n_row + (b))

    // std::vector<std::vector<double>> dist;
    double dist[n_col * n_row * n_row];
    int valid[n_col * n_row], next_valid[n_col * n_row], prev_valid[n_col * n_row];
    double f[n_row * grid_n];
    int pre[n_row * grid_n];

    double max_dist = 0;
    for (int j = 0; j < n_col; ++j) {
        for (int i = 0; i < n_row; ++i) {
            DIST(j, i, i) = 0;
            VALID(j, i) = _valid[i][j];
        }
        for (int i = 1; i < n_row; ++i) {
            DIST(j, i, i - 1) = DIST(j, i - 1, i) = _dist[i][j];
            if (_dist[i][j] > max_dist) {
                max_dist = _dist[i][j];
            }
        }
    }
    for (int i = 0, k = 1; i < n_col; ++i) {
        for (int j = 0; j + k < n_row; ++j) {
            if (DIST(i, j, j + k) == 0) {
                DIST(i, j, j + k) = DIST(i, j + k, j) = max_dist * 0.1;
            }
        }
    }
    
    for (int j = 0; j < n_row; ++j) {
        int left = 0, right = n_col - 1;
        while (left < right && VALID(left, j) == -1) {
            ++left;
        }
        while (left < right && VALID(right, j) == -1) {
            --right;
        }
        left = max(0, left - 2);
        right = min(n_col - 1, right + 2);
        for (int i = left; i <= right; ++i) {
            VALID(i, j) += 1;
        }
    }
    
    for (int i = 0; i < n_col; ++i) {
        int last_valid = -1;
        for (int j = n_row - 1; j >= 0; --j) {
            next_valid[i * n_row + j] = last_valid;
            if (VALID(i, j) >= 0) {
                last_valid = j;
            }
        }
        last_valid = 0;
        for (int j = 0; j < n_row; ++j) {
            prev_valid[i * n_row + j] = last_valid;
            if (VALID(i, j) >= 0) {
                last_valid = j;
            }
        }
        prev_valid[i * n_row] = -1;
    }

    double grid_step = 1.0 / grid_n;
    int most_valid_column = -1, max_cnt = -1;
    for (int i = 0; i < n_col; ++i) {
        int cnt = 0;
        for (int j = 0; j < n_row; ++j) {
            cnt += VALID(i, j);
        }
        if (cnt > max_cnt) {
            max_cnt = cnt;
            most_valid_column = i;
        }
    }

    for (int left = 0; left < n_row - 1; ++left) {
        if (left == 0 || labels[left] != labels[left - 1]) {
            int right = left + 1;
            while (right < n_row && labels[right] == labels[left]) ++right;
            for (int i = 0; i < n_col; ++i) {
                double totd = 0;
                for (int j = left; j < right; ++j) {
                    totd += DIST(i, j, j - 1);
                }
                if (totd > 0) {
                    double remain = totd * (1 - inner_cluster_alpha);
                    for (int j = left; j < right; ++j) {
                        DIST(i, j, j - 1) *= inner_cluster_alpha;
                        DIST(i, j - 1, j) *= inner_cluster_alpha;
                    }
                    if (left > 0) {
                        DIST(i, left, left - 1) += remain * 0.5 + max_dist * between_cluster_margin;
                        DIST(i, left - 1, left) += remain * 0.5 + max_dist * between_cluster_margin;
                    }
                    if (right < n_row) {
                        DIST(i, right, right - 1) += remain * 0.5;
                        DIST(i, right - 1, right) += remain * 0.5;
                    }
                }
            }
        }
    }

    for (int i = 0; i < n_col; ++i) {
        for (int k = 2; k < n_row; ++k) {
            for (int j = 0; j + k < n_row; ++j){
                DIST(i, j, j + k) = DIST(i, j + k, j) = DIST(i, j, j + k - 1) + DIST(i, j + k - 1, j + k);
            }
        }
    }

    max_dist = 1;
    for (int i = 0; i < n_col; ++i) {
        if (DIST(i, 0, n_row - 1) > max_dist) {
            max_dist = DIST(i, 0, n_row - 1);
        }
    }

    for (int i = 0; i < n_col; ++i) {
        for (int j = 0; j < n_row; ++j) {
            for (int k = 0; k < n_row; ++k){
                DIST(i, j, k) /= max_dist;
            }
        }
    }

    /*
    std::vector<std::pair<double, std::pair<int, int>>> all_dist;
    for (int i = 0; i < n_col; ++i) {
        for (int j = 1; j < n_row; ++j) {
            if (DIST(i, j, j - 1) > 0) {
                all_dist.push_back(std::make_pair(DIST(i, j, j - 1), std::make_pair(i, j)));
            }
        }
    }
    #ifndef NDEBUG 
    std::cout << "line 159" << std::endl;
    #endif

    std::vector<std::vector<double>> dist_quantile;
    for (int i = 0; i < n_col; ++i) {
        std::vector<double> row;
        for (int j = 0; j < n_row; ++j) {
            row.push_back(1);
        }
        dist_quantile.push_back(row);
    }
    for (unsigned i = 0, k = 0; i < all_dist.size(); ++i) {
        auto p = all_dist[i].second;
        if (i > 0 && all_dist[i].first != all_dist[i - 1].first) {
            k = i;
        }
        dist_quantile[p.first][p.second] = 1 - k / (all_dist.size() + 1);
    }
    */

    /*
    for (int i = most_valid_column + 1; i < n_col; ++i) {
        for (int j = 1; j < n_row; ++j) {
            if (DIST(i - 1, j, j - 1) > 0.1 && DIST(i - 1, j, j - 1) < DIST(i, j, j - 1)) {
                DIST(i - 1, j, j - 1) = DIST(i, j, j - 1);
            }
        }
    }
    for (int i = most_valid_column - 1; i >= 0; --i) {
        for (int j = 0; j < n_row; ++j) {
            if (DIST(i + 1, j, j - 1) > 0.1 && DIST(i + 1, j, j - 1) < DIST(i, j, j - 1)) {
                DIST(i + 1, j, j - 1) = DIST(i, j, j - 1);
            }
        }
    }
    */

    std::vector<std::vector<double>> y;
    for (int i = 0; i < n_col; ++i) {
        std::vector<double> row;
        for (int j = 0; j < n_row; ++j) {
            row.push_back(0);
        }
        y.push_back(row);
    }

    auto optimize = [&](int curr, int prev) {
        // std::cout << curr << ' ' << prev << std::endl;

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
                f[k] = sqr(t - (1 - grid_step));
            }
        }

        for (int j = 1; j < n_row; ++j) {
            for (int k = 0; k < grid_n - j * min_valid_gap; ++k) {
                double t = k * grid_step;
                double ft = 0;
                int st, ed;
                if (prev != -1) {
                    if (t == y[prev][j]) {
                        ft = -beta * sqr(grid_step);
                    } else if (VALID(prev, j) >= 0) {
                        ft = sqr(t - y[prev][j]) * alpha;
                    }
                } else {
                    ft = 0;
                }

                if (VALID(curr, j) >= 0) {
                    int h = prev_valid[curr * n_row + j];
                    double delta = DIST(curr, j, h);
                    st = min(grid_n, k + min_valid_gap);
                    ed = grid_n;
                    for (int l = st; l < ed; ++l) {
                        double d = (l - k) * grid_step;
                        double cost = f[h * grid_n + l] + ft + sqr(d - delta);// * sqr(dist_quantile[curr][j]);
                        if (cost < f[j * grid_n + k]) {
                            f[j * grid_n + k] = cost;
                            pre[j * grid_n + k] = l;
                        }
                    }
                } else {
                    f[j * grid_n + k] = f[(j - 1) * grid_n + k];
                }
            }
        }

        int k = 0, last = n_row - 1;
        if (VALID(curr, last) < 0) {
            last = prev_valid[curr * n_row + last];
        }
        for (int j = 0; j < grid_n - 1; ++j) {
            if (f[last * grid_n + j] < f[last * grid_n + k]) {
                k = j;
            }
        }

        for (int j = last; j >= 0; j = prev_valid[curr * n_row + j]) {
            // assert(curr >= 0 && curr < n_col);
            y[curr][j] = k * grid_step;
            k = pre[j * grid_n + k];
        }

        if (prev != -1) {
            for (int j = 1; j < n_row; ++j) if (VALID(curr, j) < 0) {
                // std::cout << curr << ' ' << prev << ' ' << j << ' ' << y[prev][j] << std::endl;
                y[curr][j] = y[prev][j];//, y[curr][j - 1] + grid_step);
            }
        } else {
            for (int j = 1; j < n_row; ++j) if (VALID(curr, j) < 0) {
                y[curr][j] = y[curr][j - 1] - grid_step;
            }
        }
    };

    optimize(most_valid_column, -1);

    for (int i = most_valid_column + 1; i < n_col; ++i) {
        optimize(i, i - 1);
    }

    for (int i = most_valid_column - 1; i >= 0; --i) {
        optimize(i, i + 1);
    }

    /*
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
    */
    
    std::vector<std::vector<double>> ret;
    for (int j = 0; j < n_row; ++j) {
        std::vector<double> row;
        for (int i = 0; i < n_col; ++i) {
            row.push_back(y[i][j]);
        }
        ret.push_back(row);
    }
    return ret;
    // y *= max_sum
}

PYBIND11_MODULE(DAGLayout, m) {
    m.doc() = "DAGLayout algorithm"; // optional module docstring
    m.def("DAGLayout", &DAGLayout, "A function");
}