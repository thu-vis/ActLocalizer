#include <iostream>
#include <fstream>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <omp.h>
#include <queue>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef _MSC_VER
    #include <intrin.h>
    #include <nmmintrin.h>
    #define __builtin_popcountll _mm_popcnt_u64
#endif
const double epsilon = 1e-5;
const int baseRd = 13331;
typedef unsigned long long uLL;

inline uLL exp2(int k) { return 1ull << k; }
inline uLL mask2(int k) { return (1ull << k) - 1; }

struct Node {
    int value;
    Node *next;
    Node(){}
};

struct NodePool {
    std::vector<std::vector<int>> pool;
    std::vector<int> tail;
    int ptr;
    NodePool(){}
    ~NodePool(){}
    void reset(){
        ptr = 0;
        for (int i = 0; i < tail.size(); ++i) {
            tail[i] = -1;
        }
    }
    void pop() {
        --tail[ptr];
    }
    bool empty() {
        // return ptr >= tail.size();
        while (ptr < tail.size() && tail[ptr] < 0) ++ptr;
        return ptr >= tail.size();
    }
    int minDist() {
        // while (ptr < tail.size() && tail[ptr] < 0) ++ptr;
        return ptr;
    }
    int minValue() {
        // while (ptr < tail.size() && tail[ptr] < 0) ++ptr;
        return pool[ptr][tail[ptr]];
    }
    void add(int d, int x) {
        while (d >= pool.size()) {
            pool.push_back(std::vector<int>());
            tail.push_back(-1);
        }
        if (++tail[d] >= pool[d].size()) {
            pool[d].push_back(x);
        } else {
            pool[d][tail[d]] = x;
        }
    }
};

struct pair_hash {
    inline uLL operator()(const std::pair<int,int>& p) const {
        return (uLL)(p.first) * 1333331 + p.second;
    }
};

struct ClusDist {
    double d;
    int i;
    int j;
    ClusDist(double dist, int cluster_i, int cluster_j): d(dist), i(cluster_i), j(cluster_j) {}
    bool operator<(const ClusDist& other) const {
        return d > other.d;
    }
};

class Space {
private:
    std::unordered_map<uLL, uLL> isomorphic;
    std::unordered_map<uLL, int> nodeId;
    std::vector<uLL> nodeList;
    std::vector<std::pair<int, int>>** edgeList;
    int N, M, K, LN, LM, LK;
    std::vector<std::pair<int, int>> adjIndex;
    // N is the num of nodes, M is the num of edges, and K is the num of status
    uLL maskN, maskM, maskK;
    int edgeCost;
    std::vector<std::vector<int> > nodeCost;
    std::vector<std::pair<int, int>> **allNeighbors;
    int noneCost;
    std::vector<NodePool> threadNodePool;
    std::vector<int*> distPool;
    std::vector<int*> flagPool;
    bool* inSubset;
    int threadNum;
    std::string dataName;

    std::vector<std::vector<int> > getIsomorphicDAG(uLL s) {
        std::vector<std::vector<int> > retPermutations;
        std::vector<int> permutation;
        for (int i = 0; i < N; ++i) {
            permutation.push_back(i);
        }
        while (std::next_permutation(permutation.begin() + 1, permutation.end() - 1)) {
            uLL ms = s;
            int verified_count = 0;
            for (int i = 0; i < N - 1; ++i) {
                uLL e = (ms & mask2(N - i - 1)) << i + 1;
                for (int j = i + 1; j < N - 1; ++j) if (e & exp2(j)) {
                    if (permutation[i] > permutation[j]) {
                        verified_count++;
                    }
                }
                if (verified_count > 0) break;
                ms >>= N - i - 1;
            }
            if (verified_count == 0) {
                retPermutations.push_back(std::vector<int>(permutation));
            }
        }
        return retPermutations;
    }
    uLL permuteDAG(uLL s, const std::vector<int>& permutation) {
        uLL ms = s & maskM;
        uLL ns = s >> LM;
        std::vector<int> edges;
        for (int i = 0; i < N - 1; ++i) {
            edges.push_back(0);
        }
        uLL ret = 0;
        for (int i = 0; i < N - 1; ++i) {
            uLL e = (ms & mask2(N - i - 1)) << i + 1;
            for (int j = i + 1; j < N; ++j) if (e & exp2(j)) {
                edges[permutation[i]] |= exp2(permutation[j]);
            }
            ms >>= N - i - 1;
        }
        for (int i = N - 2; i >= 1; --i) {
            ret |= ((ns >> ((i - 1) * LK)) & maskK) << ((permutation[i] - 1) * LK);
        }
        for (int i = N - 2; i >= 0; --i) {
            ret <<= N - i - 1;
            ret |= (edges[i] >> i + 1) & mask2(N - i - 1);
        }
        return ret;
    }
    std::vector<std::pair<int, int>>* findOneStep(uLL s) {
        uLL ms = s & maskM;
        uLL ns = s >> LM;
        auto ret = new std::vector<std::pair<int, int>>();
        for (int i = 0; i < LM; ++i) {
            uLL ms0 = ms ^ exp2(i);
            // if (__builtin_popcountll(ms0) > M) continue;
            auto s0 = (ns << LM) + ms0;
            if (isLegal(s0)) {
                auto it = isomorphic.find(s0);
                if (it == isomorphic.end()) continue;
                int id = nodeId.find(it->second)->second;
                ret->push_back(std::make_pair(id, edgeCost));
            } else {
                bool has_i = (ms & exp2(i)) > 0;
                if (ms & exp2(i)) {
                    for (int j = i + 1; j < LM; ++j) if (adjIndex[i].second == adjIndex[j].first && (ms & exp2(j))) {
                        ms0 = ms ^ exp2(i) ^ exp2(j);
                        int k1 = (ns >> ((adjIndex[i].second - 1) * LK)) & maskK;
                        auto ns0 = ns - (k1 << ((adjIndex[i].second - 1) * LK));
                        s0 = (ns0 << LM) + ms0;
                        if (isLegal(s0)) {
                            auto it = isomorphic.find(s0);
                            if (it == isomorphic.end()) continue;
                            int id = nodeId.find(it->second)->second;
                            ret->push_back(std::make_pair(id, edgeCost * 2 + noneCost));
                        }
                    }
                } else if (__builtin_popcountll(ms) + 2 <= M) {
                    for (int j = i + 1; j < LM; ++j) if (adjIndex[i].second == adjIndex[j].first && !(ms & exp2(j))) {
                        ms0 = ms ^ exp2(i) ^ exp2(j);
                        int k1 = (ns >> ((adjIndex[i].second - 1) * LK)) & maskK;
                        for (int k2 = 0; k2 < K; ++k2) {
                            auto ns0 = ns - (k1 << ((adjIndex[i].second - 1) * LK)) + (k2 << ((adjIndex[i].second - 1) * LK));
                            s0 = (ns0 << LM) + ms0;
                            if (isLegal(s0)) {
                                auto it = isomorphic.find(s0);
                                if (it == isomorphic.end()) continue;
                                int id = nodeId.find(it->second)->second;
                                ret->push_back(std::make_pair(id, edgeCost * 2 + noneCost));
                            }
                        }
                    }
                }
                for (int j = i + 1; j < LM; ++j) if (adjIndex[i].second == adjIndex[j].second || adjIndex[i].first == adjIndex[j].first) {
                    bool has_j = (ms & exp2(j)) > 0;
                    if (has_i == has_j) continue;
                    ms0 = ms ^ exp2(i) ^ exp2(j);
                    auto ns0 = ns;
                    s0 = (ns0 << LM) + ms0;
                    if (isLegal(s0)) {
                        auto it = isomorphic.find(s0);
                        if (it == isomorphic.end()) continue;
                        int id = nodeId.find(it->second)->second;
                        ret->push_back(std::make_pair(id, edgeCost * 2));
                    }
                }
            }
        }
        for (int i = 0; i < N - 2; ++i) {
            int k1 = (ns >> (i * LK)) & maskK;
            for (int k2 = 0; k2 < K; ++k2) if (k1 != k2) {
                auto ns0 = ns - (k1 << (i * LK)) + (k2 << (i * LK));
                auto s0 = (ns0 << LM) + ms;
                if (!isLegal(s0)) continue;
                auto it = isomorphic.find(s0);
                if (it == isomorphic.end()) continue;
                int id = nodeId.find(it->second)->second;
                ret->push_back(std::make_pair(id, nodeCost[k1][k2]));
            }
        }
        return ret;
    }
    void init_graph(int _N, int _M, int _K) {
        N = _N;
        M = _M;
        K = _K;
        for (LK = 1; (1 << LK) < K; ++LK);
        LN = (N - 2) * LK;
        LM = N * (N - 1) >> 1;
        maskM = exp2(LM) - 1;
        maskN = exp2(LN) - 1;
        maskK = exp2(LK) - 1;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                adjIndex.push_back(std::make_pair(i, j));
            }
        }
    }
public:
    Space(){
        threadNum = omp_get_num_procs();
        std::cout << "Thread Num: " << threadNum << std::endl;
        for (int i = 0; i < threadNum; ++i) {
            threadNodePool.push_back(NodePool());
        }
        allNeighbors = NULL;
        inSubset = NULL;
    }
    ~Space(){
        for (int i = 0; i < nodeList.size(); ++i) {
            delete edgeList[i];
        }
        for (int i = 0; i < threadNum; ++i) {
            delete distPool[i];
            delete flagPool[i];
        }
        if (inSubset != NULL) {
            delete inSubset;
        }
        if (allNeighbors != NULL) {
            auto n = numOfDAG();
            for (int i = 0; i < n; ++i) {
                delete allNeighbors[i];
            }
            delete allNeighbors;
        }
        delete edgeList;
    }
    int numOfDAG() {
        return nodeList.size();
    }
    std::vector<uLL> allDAG() {
        return nodeList;
    }
    bool isLegal(uLL s) {
        uLL ms = s & maskM;
        uLL ns = s >> LM;
        // the num of edges should be less than M.
        if (__builtin_popcountll(ms) > M) {
            return false;
        }
        uLL curr = 1;
        for (int i = 0; i < N - 1; ++i) {
            uLL e = (ms & mask2(N - i - 1)) << i + 1;
            if (!(curr & exp2(i))) {
                // disconnected node cannot have out edges.
                if (e > 0){ return false; }
                // the value of disconnected node should be zero.
                if (i > 0 && (ns & maskK)) {
                    return false;
                }
            } else if (!e) {
                return false;
            }
            curr |= e;
            ms >>= N - i - 1;
            if (i > 0) {
                ns >>= LK;
            }
        }
        if (!(curr & exp2(N - 1))) {
            return false;
        }
        return true;
    }
    void init301(const std::vector<std::vector<int>> & dist) {
        int n = dist.size();
        for (int k = 0; k < n; ++k) {
            nodeList.push_back(k);
            nodeId[k] = k;
        }
        for (int i = 0; i < threadNum; ++i) {
            distPool.push_back(new int[nodeList.size()]);
            flagPool.push_back(new int[nodeList.size()]);
        }
        inSubset = new bool[nodeList.size()];

        edgeList = new std::vector<std::pair<int, int>>*[nodeList.size()];
        #pragma omp parallel for
        for (int k = 0; k < n; ++k) {
            // std::cout << k << " processing" << std::endl;
            int threadId = omp_get_thread_num();
            auto flag = flagPool[threadId];
            int currFlag = k + 1 + baseRd * (rand() & 0xffff);
            flag[k] = currFlag;
            auto d = dist[k];
            auto ret = new std::vector<std::pair<int, int>>();
            for (int i = 1; i < n; ++i) {
                int min_dist = 1e7, min_idx = -1;
                for (int j = 0; j < n; ++j) {
                    if (flag[j] == currFlag) continue;
                    if (d[j] < min_dist) {
                        min_idx = j;
                        min_dist = d[j];
                    }
                }
                ret->push_back(std::make_pair(min_idx, min_dist));
                flag[min_idx] = currFlag;
            }
            edgeList[k] = ret;
        }
        
        int count = 0;
        for (int i = 0; i < nodeList.size(); ++i) {
            count += edgeList[i]->size();
        }
    }

    std::vector<std::pair<int, int>>* findSingleSourceDistInSubset(const std::vector<int> &idx, int src) {
        // try {
        int threadId = omp_get_thread_num();
        auto nodePool = &threadNodePool[threadId];
        auto dist = distPool[threadId];
        auto flag = flagPool[threadId];
        int currFlag = src + 1 + baseRd * (rand() & 0xffff);
        nodePool->reset();
        //std::unordered_map<int, int> dist;
        nodePool->add(0, src);
        dist[src] = 0;
        flag[src] = currFlag;
        int count = 0;
        std::vector<std::pair<int, int>>* ret = new std::vector<std::pair<int, int>>();
        // std::cout << "start " << src << std::endl;
        while (count < idx.size()) {
            if (nodePool->empty()) {
                break;
            }
            int d = nodePool->minDist();
            int x = nodePool->minValue();
            nodePool->pop();
            if (d > dist[x]) continue;
            if (inSubset[x]) {
                count += 1;
            }
            for (auto e: *edgeList[x]) {
                if (flag[e.first] != currFlag) {
                    flag[e.first] = currFlag;
                    dist[e.first] = d + e.second;
                    nodePool->add(d + e.second, e.first);
                } else if (dist[e.first] > d + e.second) {
                    dist[e.first] = d + e.second;
                    nodePool->add(d + e.second, e.first);
                }
            }
        }
        // std::cout << "inter " << src << std::endl;
        for (auto i: idx) {
            ret->push_back(std::make_pair(i, dist[i]));
        }
        // std::cout << "done " << src << std::endl;
        std::cout << "";
        return ret;
        // }
        // catch (std::exception &e) {
        //     std::cout << "errrrrrrrrrrrrrrrrrror" << std::endl;
        //     std::cout << e.what() << std::endl;
        //     std::cerr << e.what() << std::endl;
        //     std::cerr << typeid(e).name() << std::endl;
        // }
    }   

    std::vector<std::pair<int, int>>* findNeighborsInSubset(const std::vector<int> &idx, int src, int k = -1) {
        int threadId = omp_get_thread_num();
        //std::cout << "threadId" << threadId << std::endl;
        auto nodePool = &threadNodePool[threadId];
        auto dist = distPool[threadId];
        auto flag = flagPool[threadId];
        int currFlag = src + 1 + baseRd * (rand() & 0xffff);
        nodePool->reset();
        //std::unordered_map<int, int> dist;
        nodePool->add(0, src);
        dist[src] = 0;
        flag[src] = currFlag;
        int top = 0;
        auto ret = new std::vector<std::pair<int, int>>();
        if (k == -1 || k > idx.size()) {
            k = idx.size();
        }
        // std::cout <<< "ckpt" << std::endl;
        while (ret->size() < k) {
            if (nodePool->empty()) {
                break;
            }
            int d = nodePool->minDist();
            int x = nodePool->minValue();
            nodePool->pop();
            if (d > dist[x]) continue;
            if (inSubset[x]) {
                ret->push_back(std::make_pair(x, d));
            }
            for (auto e: *edgeList[x]) {
                if (flag[e.first] != currFlag) {
                    flag[e.first] = currFlag;
                    dist[e.first] = d + e.second;
                    nodePool->add(d + e.second, e.first);
                } else if (dist[e.first] > d + e.second) {
                    dist[e.first] = d + e.second;
                    nodePool->add(d + e.second, e.first);
                }
            }
        }
        std::cout << "";
        //std::cout << "size #" << src << " " << ret->size() << std::endl;
        return ret;
    }

    std::vector<std::pair<int, int>>* findNeighbors(int src, int k = -1) {
        int threadId = omp_get_thread_num();
        auto nodePool = &threadNodePool[threadId];
        auto dist = distPool[threadId];
        auto flag = flagPool[threadId];
        int currFlag = src + 1 + baseRd * (rand() & 0xffff);
        nodePool->reset();
        //std::unordered_map<int, int> dist;
        nodePool->add(0, src);
        dist[src] = 0;
        flag[src] = currFlag;
        int top = 0;
        auto ret = new std::vector<std::pair<int, int>>();
        if (k == -1) {
            k = nodeList.size();
        }
        while (ret->size() <  k) {
            if (nodePool->empty()) {
                break;
            }
            int d = nodePool->minDist();
            int x = nodePool->minValue();
            nodePool->pop();
            if (d > dist[x]) continue;
            ret->push_back(std::make_pair(x, d));
            for (auto e: *edgeList[x]) {
                if (flag[e.first] != currFlag) {
                    flag[e.first] = currFlag;
                    dist[e.first] = d + e.second;
                    nodePool->add(d + e.second, e.first);
                } else if (dist[e.first] > d + e.second) {
                    dist[e.first] = d + e.second;
                    nodePool->add(d + e.second, e.first);
                }
            }
        }
        // std::cout << "size #" << src << " " << ret->size() << std::endl;
        return ret;
    }
    
    void calcNeighbors(int k) {
        auto n = numOfDAG();
        if (allNeighbors != NULL) {
            for (int i = 0; i < n; ++i) {
                delete allNeighbors[i];
            }
        } else {
            allNeighbors = new std::vector<std::pair<int, int>>*[n];
        }
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            allNeighbors[i] = findNeighbors(i, k);
        }
    }

    void calcNeighborsInSubset(const std::vector<int>& idx, int k) {
        auto n = numOfDAG();
        memset(inSubset, 0, sizeof(bool) * n);
        for (auto x: idx) {
            inSubset[x] = 1;
        }
        if (allNeighbors != NULL) {
            for (auto i: idx) {
                if (allNeighbors[i] != NULL) {
                    delete allNeighbors[i];
                }
            }
        } else {
            allNeighbors = new std::vector<std::pair<int, int>>*[numOfDAG()];
            memset(allNeighbors, 0, sizeof(std::vector<std::pair<int, int>>*) * n);
        }
        #pragma omp parallel for
        for (int j = 0; j < idx.size(); ++j) {
            int i = idx[j];
            allNeighbors[i] = findNeighborsInSubset(idx, i, k);
        }
    }

    std::vector<std::vector<int>> getDistMatrix(std::vector<int> idx) {
        int n = idx.size();
        std::vector<std::pair<int,int>>** retPlaceho = new std::vector<std::pair<int,int>>*[n];
        memset(inSubset, 0, sizeof(bool) * numOfDAG());
        for (auto x: idx) {
            inSubset[x] = 1;
        }
        std::cout << "length: " << idx.size() << std::endl;
        // std::cout << "----1" << std::endl;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            // std::cout << idx[i] << std::endl;
            // std::cout << i << std::endl;
            // retPlaceho[i] = findNeighbors(idx[i]);
            // std::cout << "sort" << idx[i] << std::endl;
            // std::sort(retPlaceho[i]->begin(), retPlaceho[i]->end());
            retPlaceho[i] = findSingleSourceDistInSubset(idx, idx[i]);
            // for (int j = 1; j < retPlaceho[i]->size(); ++j) {
            //     if (retPlaceho[i]->at(j - 1).first + 1 != retPlaceho[i]->at(j).first) {
            //         for (int k = retPlaceho[i]->at(j - 1).first + 1; k < retPlaceho[i]->at(j).first; ++k)
            //             std::cout << k << ' ' << interpreter(k) << std::endl;
            //     }
            // }
            // std::cout << "sorted" << i << std::endl;
        }
        // std::cout << "----2" << std::endl;
        std::vector<std::vector<int>> ret;
        for (int i = 0; i < n; ++i) {
            std::vector<int> rows;
            for (int j = 0; j < n; ++j) {
                // rows.push_back(retPlaceho[i]->at(idx[j]).second);
                rows.push_back(retPlaceho[i]->at(j).second);
            }
            ret.push_back(rows);
        }
        // std::cout << "----3" << std::endl;
        for (int i = 0; i < n; ++i) {
            delete retPlaceho[i];
        }
        delete retPlaceho;
        return ret;
    }

    std::vector<std::vector<int>> meanshiftInSubset(const std::vector<int>& idx, int band_width, int influence_width, bool is_epsilon = false) {
        auto n = numOfDAG();
        memset(inSubset, 0, sizeof(bool) * n);
        for (auto x: idx) {
            inSubset[x] = 1;
        }
        if (!is_epsilon) {
            // calcNeighborsInSubset(idx, influence_width);
            calcNeighborsInSubset(idx, band_width);
            influence_width -= 1;
            band_width -= 1;
        } else {
            calcNeighborsInSubset(idx, (int)(100 + sqrt(idx.size())));
        }
        std::vector<std::vector<int>> ret;
        int *next = new int[n],
            *degree = new int[n],
            *Q = new int[n];
        memset(next, 0, sizeof(int) * n);

        if (band_width < 0) {
            band_width = int(sqrt(idx.size()));
            std::cout << "bw:" << band_width << std::endl;
        }

        #pragma omp parallel for
        for (int k = 0; k < idx.size(); ++k) {
            int i = idx[k];
            int threadId = omp_get_thread_num();
            auto flag = flagPool[threadId];
            auto dist = distPool[threadId];
            int currFlag = i + 1 + baseRd * (rand() & 0xffff);
            int index = 0;
            int minCount = 0;
            float minDist = 0, w;
            int neighbor_r = 0;
            index = 0;
            for (auto x: *allNeighbors[i]) {
                ++index;
                if (is_epsilon && x.second > band_width || !is_epsilon && index > band_width) {
                    break;
                }
                minCount++;
                dist[x.first] = x.second; 
                // w = exp(-pow(float(dist[x.first])/h[x.first], 2)/2);
                w = exp(-pow(float(dist[x.first])/band_width, 2)/2);
                minDist += w * pow(x.second, 2);
                flag[x.first] = currFlag;
                neighbor_r = x.second;
            }
            next[i] = -1;
            index = 0;
            for (auto e: *allNeighbors[i]) {
                if (is_epsilon && e.second > band_width || !is_epsilon && index > band_width) {
                    break;
                }
                int newCount = 0;
                int lastEdge = 0;
                float newDist = 0;
                for (auto x: *allNeighbors[e.first]) {
                    if (flag[x.first] == currFlag) {
                        newCount++;
                        // w = exp(-pow(float(dist[x.first])/h[x.first], 2)/2);
                        w = exp(-pow(float(dist[x.first])/band_width, 2)/2);
                        newDist += w * pow(x.second, 2);
                        if (newCount == minCount) break;
                    } 
                    else {
                        // w = exp(-pow(float(x.second + e.second)/h[x.first], 2)/2);
                        w = exp(-pow(float(x.second + e.second)/band_width, 2)/2);
                        newDist += w * pow(x.second, 2);
                    }
                    lastEdge = x.second;
                }

                if (newDist < minDist) {
                    next[i] = e.first;
                    minDist = newDist;
                }
            }
        }

        int *tmp_label = new int[n];
        memset(tmp_label, 0, sizeof(int) * n);
        int numLables = 0;
        std::vector<int> tmp;
        int *visited = flagPool[0];
        int *isRoot = distPool[0];
        memset(visited, 0, sizeof(int) * n);
        memset(isRoot, 0, sizeof(int) * n);
        for (auto i: idx) {
            if (tmp_label[i] == 0) {
                tmp.clear();
                int j = i, currLabel, currFlag = i + 1 + baseRd * (rand() & 0xffff);
                for (; next[j] != -1 && visited[j] != currFlag; j = next[j]) {
                    visited[j] = currFlag;
                    tmp.push_back(j);
                }
                visited[j] = currFlag;
                tmp.push_back(j);
                int index = 0;
                if (tmp_label[j] == 0) {
                    for (auto x: *allNeighbors[j]) {
                        ++index;
                        if (is_epsilon && x.second > band_width || !is_epsilon && index > band_width) {
                            break;
                        }
                        if (tmp_label[x.first] > 0) {
                            j = x.first;
                            break;
                        }
                    }
                }
                if (tmp_label[j] == 0) {
                    ++numLables;
                    currLabel = numLables;
                    next[j] = -1;
                    isRoot[j] = 1;
                    tmp_label[j] = currLabel;
                } else {
                    currLabel = tmp_label[j];
                }
                for (int x: tmp) {
                    tmp_label[x] = currLabel;
                }
            }
        }

        int *label = Q;
        memset(label, 0, sizeof(int) * n);
        #pragma omp parallel for
        for (int j = 0; j < idx.size(); ++j) {
            int i = idx[j];
            int threadId = omp_get_thread_num();
            auto count = distPool[threadId];
            memset(count, 0, sizeof(int) * (numLables + 1));
            label[i] = tmp_label[i];
        }

        std::cout << "num of clusters: " << numLables << std::endl;
        for (int currlabel = 1; currlabel <= numLables; ++currlabel) {
            int count = 0;
            std::vector<int> cluster;
            for (auto i: idx) {
                if (label[i] == currlabel) {
                    if (isRoot[i]) {
                        if (cluster.size() > 0) {
                            cluster.push_back(cluster[0]);
                            cluster[0] = i;
                        } else {
                            cluster.push_back(i);
                        }
                    } else {
                        cluster.push_back(i);
                    }
                }
            }
            ret.push_back(cluster);
        }

        std::cout << "cluster done" << std::endl;

        delete tmp_label;
        delete next;
        delete Q;
        delete degree;
        return ret;
    }

    std::vector<std::vector<int>> meanshift(int band_width, int influence_width, bool is_epsilon = false) {
        if (!is_epsilon) {
            calcNeighbors(band_width);
            influence_width -= 1;
            band_width -= 1;
        } else {
            calcNeighbors((int)(200 + sqrt(numOfDAG())));
            int nn_reach = 0;
            for (int i = 0; i < numOfDAG(); ++i) {
                if (allNeighbors[i]->back().second > band_width) {
                    ++nn_reach;
                }
            }
            std::cout << "nn_reach " << nn_reach << std::endl;
        }
        std::vector<std::vector<int>> ret;
        auto n = numOfDAG();
        int *next = new int[n],
            *degree = new int[n],
            *Q = new int[n];
        memset(next, 0, sizeof(int) * n);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            int threadId = omp_get_thread_num();
            auto flag = flagPool[threadId];
            auto dist = distPool[threadId];
            int currFlag = i + 1 + baseRd * (rand() & 0xffff);
            int index = 0;
            // int minDist = 0;
            int minCount = 0;
            float minDist = 0, w;
            int neighbor_r = 0;
            index = 0;
            for (auto x: *allNeighbors[i]) {
                ++index;
                if (is_epsilon && x.second > band_width || !is_epsilon && index > band_width) {
                    break;
                }
                // minDist += x.second;
                minCount++;
                dist[x.first] = x.second; 
                // w = exp(-pow(float(dist[x.first])/band_width, 2)/2);
                w = 1;
                minDist += w * pow(x.second, 2);
                flag[x.first] = currFlag;
                neighbor_r = x.second;
            }
            next[i] = -1;
            index = 0;
            // for (auto e: *edgeList[i]) {
            for (auto e: *allNeighbors[i]) {
                ++index;
                if (is_epsilon && e.second > band_width || !is_epsilon && index > band_width) {
                    break;
                }
                // int newDist = 0;
                int newCount = 0;
                int lastEdge = 0;
                float newDist = 0;
                for (auto x: *allNeighbors[e.first]) {
                    if (x.first == i) continue;
                    if (flag[x.first] == currFlag) {
                        // newDist += x.second;
                        newCount++;
                        // w = exp(-pow(float(dist[x.first])/band_width, 2)/2);
                        w = exp(-pow(float(dist[x.first])/influence_width, 2)/2);
                        // w = 1;
                        newDist += w * pow(x.second, 2);
                        if (newCount == minCount) break;
                    } 
                    else {
                        // w = exp(-pow(float(x.second + band_width)/band_width, 2)/2);
                        w = exp(-pow(float(x.second + influence_width)/influence_width, 2)/2);
                        newDist += w * pow(x.second, 2);
                    }
                    lastEdge = x.second;
                }
                // newDist += (minCount - newCount) * pow(lastEdge, 2);
                if (newDist < minDist) {
                    next[i] = e.first;
                    minDist = newDist;
                }
                // if (newDensity > maxDensity) {
                //     next[i] = e.first;
                //     maxDensity = newDensity;
                // }
            }
        }

        int sum = 0;
        for (int i = 0; i < n; ++i) {
            if (next[i] == -1) {
                sum++;
            }
        }
        std::cout << sum << "===" << numOfDAG() - sum << std::endl;
        
        int *tmp_label = new int[n];
        memset(tmp_label, 0, sizeof(int) * n);
        int numLables = 0; 
        std::vector<int> tmp;
        int *visited = flagPool[0];
        int *isRoot = distPool[0];
        memset(visited, 0, sizeof(int) * n);
        memset(isRoot, 0, sizeof(int) * n);
        for (int i = 0; i < n; ++i) {
            if (tmp_label[i] == 0) {
                tmp.clear();
                int j = i, currLabel, currFlag = i + 1 + baseRd * (rand() & 0xffff);
                for (; next[j] != -1 && visited[j] != currFlag; j = next[j]) {
                    visited[j] = currFlag;
                    tmp.push_back(j);
                }
                visited[j] = currFlag;
                tmp.push_back(j);
                int index = 0;
                if (tmp_label[j] == 0) {
                    for (auto x: *allNeighbors[j]) {
                        ++index;
                        if (is_epsilon && x.second > band_width || !is_epsilon && index > band_width) {
                            break;
                        }
                        // if (next[x.first] == -1 && label[x.first] > 0) {
                        if (tmp_label[x.first] > 0) {
                            next[j] = x.first;
                            j = x.first;
                            break;
                        }
                    }
                }
                if (tmp_label[j] == 0) {
                    ++numLables;
                    currLabel = numLables;
                    next[j] = -1;
                    isRoot[j] = 1;
                    tmp_label[j] = currLabel;
                } else {
                    currLabel = tmp_label[j];
                }
                for (int x: tmp) {
                    tmp_label[x] = currLabel;
                }
            }
        }

        int *label = Q;
        memset(label, 0, sizeof(int) * n);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            int threadId = omp_get_thread_num();
            auto count = distPool[threadId];
            memset(count, 0, sizeof(int) * (numLables + 1));
            // if (!isRoot[i]) {
            if (false) {
                int index = 0;
                for (auto x: *allNeighbors[i]) {
                    ++index;
                    if (is_epsilon && x.second > influence_width || !is_epsilon && index > influence_width) {
                        break;
                    }
                    // if (visited[x.first]) {
                        ++count[tmp_label[x.first]];
                    // }
                    int maxCount = 0;
                    for (int currlabel = 1; currlabel <= numLables; ++currlabel) {
                        if (count[currlabel] > maxCount) {
                            label[i] = currlabel;
                            maxCount = count[currlabel];
                        }
                    }
                }
            } else {
                label[i] = tmp_label[i];
            }
        }

        std::cout << "num of clusters: " << numLables << std::endl;
        for (int currlabel = 1; currlabel <= numLables; ++currlabel) {
            int count = 0;
            std::vector<int> cluster;
            for (int i = 0; i < n; ++i) {
                if (label[i] == currlabel) {
                    if (isRoot[i]) {
                        if (cluster.size() > 0) {
                            cluster.push_back(cluster[0]);
                            cluster[0] = i;
                        } else {
                            cluster.push_back(i);
                        }
                    } else {
                        cluster.push_back(i);
                    }
                }
            }
            if (cluster.size() > 0) {
                ret.push_back(cluster);
            }
        }

        delete tmp_label;
        delete next;
        delete Q;
        delete degree;
        return ret;
    }

    std::vector<std::vector<int>> meanshiftInSubsetAdaptive(const std::vector<int>& idx, float band_width, float c) {
        auto n = numOfDAG();
        memset(inSubset, 0, sizeof(bool) * n);
        for (auto x: idx) {
            inSubset[x] = 1;
        }

        calcNeighborsInSubset(idx, idx.size());


        std::vector<std::vector<int>> ret;
        int *next = new int[n],
            *degree = new int[n],
            *Q = new int[n];
        memset(next, 0, sizeof(int) * n);

        #pragma omp parallel for
        for (int k = 0; k < idx.size(); ++k) {
            int i = idx[k];
            int threadId = omp_get_thread_num();
            auto flag = flagPool[threadId];
            auto dist = distPool[threadId];
            int currFlag = i + 1 + baseRd * (rand() & 0xffff);
            int index = 0;
            int minCount = 0;
            float minDist = 0, w;
            int neighbor_r = 0;
            index = 0;
            for (auto x: *allNeighbors[i]) {
                minCount++;
                dist[x.first] = x.second; 
                w = exp(-pow(float(dist[x.first])/band_width, 2)/2);
                minDist += w * pow(x.second, 2);
                flag[x.first] = currFlag;
                neighbor_r = x.second;
            }
            // std::cout << i << " " << minDist << std::endl;
            next[i] = -1;
            index = 0;
            for (auto e: *allNeighbors[i]) {
                int newCount = 0;
                int lastEdge = 0;
                float newDist = 0;
                for (auto x: *allNeighbors[e.first]) {
                    if (flag[x.first] == currFlag) {
                        newCount++;
                        w = exp(-pow(float(dist[x.first])/band_width, 2)/2);
                        newDist += w * pow(x.second, 2);
                        if (newCount == minCount) break;
                    } 
                    else {
                        w = exp(-pow(float(x.second + e.second)/band_width, 2)/2);
                        newDist += w * pow(x.second, 2);
                    }
                    lastEdge = x.second;
                }

                if (newDist < minDist) {
                    next[i] = e.first;
                    minDist = newDist;
                }
            }
        }

        int sum = 0;
        for (int i = 0; i < n; ++i) {
            if (next[i] == -1) {
                sum++;
            }
        }
        // std::cout << "To other nodes: " << idx.size() - sum << std::endl;

        int *tmp_label = new int[n];
        memset(tmp_label, 0, sizeof(int) * n);
        int numLables = 0;
        std::vector<int> tmp;
        int *visited = flagPool[0];
        int *isRoot = distPool[0];
        memset(visited, 0, sizeof(int) * n);
        memset(isRoot, 0, sizeof(int) * n);
        for (auto i: idx) {
            if (tmp_label[i] == 0) {
                tmp.clear();
                int j = i, currLabel = 0, currFlag = i + 1 + baseRd * (rand() & 0xffff);
                for (; next[j] != -1 && visited[j] != currFlag; j = next[j]) {
                    visited[j] = currFlag;
                    tmp.push_back(j);
                }
                visited[j] = currFlag;
                tmp.push_back(j);
                int index = 0;
                if (tmp_label[j] == 0) {
                    isRoot[j] = 1;
                    for (auto x: *allNeighbors[j]) {
                        if (x.second > band_width) {
                            break;
                        }
                        if (tmp_label[x.first] > 0 && isRoot[x.first] > 0) {
                            j = x.first;
                            break;
                        }
                    }
                }
                if (tmp_label[j] == 0) {
                    ++numLables;
                    currLabel = numLables;
                    next[j] = -1;
                    tmp_label[j] = currLabel;
                } else {
                    currLabel = tmp_label[j];
                }
                for (int x: tmp) {
                    tmp_label[x] = currLabel;
                }
            }
        }

        int *label = Q;
        memset(label, 0, sizeof(int) * n);
        #pragma omp parallel for
        for (int j = 0; j < idx.size(); ++j) {
            int i = idx[j];
            int threadId = omp_get_thread_num();
            auto count = distPool[threadId];
            memset(count, 0, sizeof(int) * (numLables + 1));
            label[i] = tmp_label[i];
        }

        // std::cout << "num of clusters: " << numLables << std::endl;
        for (int currlabel = 1; currlabel <= numLables; ++currlabel) {
            int count = 0;
            std::vector<int> cluster;
            for (auto i: idx) {
                if (label[i] == currlabel) {
                    if (isRoot[i] == 1) {
                        if (cluster.size() > 0) {
                            cluster.push_back(cluster[0]);
                            cluster[0] = i;
                        } else {
                            cluster.push_back(i);
                        }
                    } else {
                        cluster.push_back(i);
                    }
                }
            }
            ret.push_back(cluster);
            // std::cout << "size:" << currlabel << "->" << cluster.size() << std::endl;
        }

        // std::cout << "cluster done" << std::endl;

        // std::cout << "Next:";
        // for(int i = 0;i < n;i++){
        //     std::cout << next[i] << " ";
        // }
        // std::cout << "Label:";
        // for(int i = 0;i < n;i++){
        //     std::cout << label[i] << " ";
        // }

        delete tmp_label;
        delete next;
        delete Q;
        delete degree;
        return ret;
    }

    std::vector<std::vector<int>> meanshiftAdaptive(float band_width, float c) {
        std::vector<std::vector<int>> ret;
        auto n = numOfDAG();

        calcNeighbors(n);

        int *next = new int[n],
            *degree = new int[n],
            *Q = new int[n];
        memset(next, 0, sizeof(int) * n);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            int threadId = omp_get_thread_num();
            auto flag = flagPool[threadId];
            auto dist = distPool[threadId];
            int currFlag = i + 1 + baseRd * (rand() & 0xffff);
            int index = 0;
            int minCount = 0;
            float minDist = 0, w;
            int neighbor_r = 0;
            index = 0;
            for (auto x: *allNeighbors[i]) {
                minCount++;
                dist[x.first] = x.second; 
                w = exp(-pow(float(dist[x.first])/band_width, 2)/2);
                // w = 1;
                minDist += w * pow(x.second, 2);
                flag[x.first] = currFlag;
                neighbor_r = x.second;
            }
            next[i] = -1;
            index = 0;
            for (auto e: *allNeighbors[i]) {
                int newCount = 0;
                int lastEdge = 0;
                float newDist = 0;
                for (auto x: *allNeighbors[e.first]) {
                    // if (x.first == i) continue;
                    if (flag[x.first] == currFlag) {
                        newCount++;
                        w = exp(-pow(float(dist[x.first])/band_width, 2)/2);
                        newDist += w * pow(x.second, 2);
                        if (newCount == minCount) break;
                    } 
                    else {
                        w = exp(-pow(float(x.second + e.second)/band_width, 2)/2);
                        newDist += w * pow(x.second, 2);
                    }
                    lastEdge = x.second;
                }
                if (newDist < minDist) {
                    next[i] = e.first;
                    minDist = newDist;
                }
            }
        }

        int sum = 0;
        for (int i = 0; i < n; ++i) {
            if (next[i] == -1) {
                sum++;
            }
        }
        // std::cout << "To other nodes: " << numOfDAG() - sum << std::endl;
        
        int *tmp_label = new int[n];
        memset(tmp_label, 0, sizeof(int) * n);
        int numLables = 0; 
        std::vector<int> tmp;
        int *visited = flagPool[0];
        int *isRoot = distPool[0];
        memset(visited, 0, sizeof(int) * n);
        memset(isRoot, 0, sizeof(int) * n);
        for (int i = 0; i < n; ++i) {
            if (tmp_label[i] == 0) {
                tmp.clear();
                int j = i, currLabel, currFlag = i + 1 + baseRd * (rand() & 0xffff);
                for (; next[j] != -1 && visited[j] != currFlag; j = next[j]) {
                    visited[j] = currFlag;
                    tmp.push_back(j);
                }
                visited[j] = currFlag;
                tmp.push_back(j);
                int index = 0;
                if (tmp_label[j] == 0) {
                    for (auto x: *allNeighbors[j]) {
                        if (x.second > band_width) {
                            break;
                        }
                        if (tmp_label[x.first] > 0 && isRoot[x.first] > 0) {
                            j = x.first;
                            break;
                        }
                    }
                }
                if (tmp_label[j] == 0) {
                    ++numLables;
                    currLabel = numLables;
                    next[j] = -1;
                    isRoot[j] = 1;
                    tmp_label[j] = currLabel;
                } else {
                    currLabel = tmp_label[j];
                    isRoot[j] = 2;
                }
                for (int x: tmp) {
                    tmp_label[x] = currLabel;
                }
            }
        }

        int *label = Q;
        memset(label, 0, sizeof(int) * n);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            int threadId = omp_get_thread_num();
            auto count = distPool[threadId];
            memset(count, 0, sizeof(int) * (numLables + 1));           
            label[i] = tmp_label[i];
        }

        // std::cout << "num of clusters: " << numLables << std::endl;
        for (int currlabel = 1; currlabel <= numLables; ++currlabel) {
            // if (getfa(fa, currlabel) != currlabel) {
            //     continue;
            // }
            int count = 0;
            std::vector<int> cluster;
            for (int i = 0; i < n; ++i) {
                if (label[i] == currlabel) {
                // if (getfa(fa, label[i]) == currlabel) {
                    if (isRoot[i] == 1) {
                    // if (isRoot[i] == 1 && label[i] == currlabel) {
                        if (cluster.size() > 0) {
                            cluster.push_back(cluster[0]);
                            cluster[0] = i;
                        } else {
                            cluster.push_back(i);
                        }
                    } else {
                        cluster.push_back(i);
                    }
                }
            }
            ret.push_back(cluster);
        }

        // std::cout << "cluster done" << std::endl;

        // std::cout << "Next:";
        // for(int i = 0;i < n;i++){
        //     std::cout << next[i] << " ";
        // }

        delete tmp_label;
        delete next;
        delete Q;
        delete degree;
        return ret;
    }
};

PYBIND11_MODULE(DAGSpace, m){
    m.doc() = "pybind11 example";
    pybind11::class_<Space>(m, "Space")
        .def( pybind11::init() )
        .def( "init", &Space::init301 )
        .def( "numOfDAG", &Space::numOfDAG )
        .def( "meanshift", &Space::meanshift )
        .def( "meanshiftAdaptive", &Space::meanshiftAdaptive )
        .def( "meanshiftInSubset", &Space::meanshiftInSubset )
        .def( "meanshiftInSubsetAdaptive", &Space::meanshiftInSubsetAdaptive )
        .def( "getDist", &Space::getDistMatrix );
}