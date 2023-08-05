import numpy as np
from .subset_selection.greedy import Greedy
from .subset_selection.ds3 import DS3
from sklearn.metrics import pairwise_distances
from .alignment import Line

def ds3_find_frames(dist, n):
    # calculate reg
    l_star = dist.sum(axis=1).argmin()
    l_reg = np.asarray([np.linalg.norm(dist[i] - dist[l_star], ord=1) for i in range(len(dist))]).max() / 2
    if n >= dist.shape[0]:
        return list(range(dist.shape[0]))
    
    def select_by_DS3(reg):
        subset_selector_solver = DS3(dist, reg)
        frames = subset_selector_solver.ADMM(1e-1, 1e-5, 1e4, np.inf, [], None, None)[-2]
        # print("[n = {} reg = {}]: \t".format(n, reg), frames)
        return frames

    # try reg / n
    frames = select_by_DS3(l_reg / n)
    nf = len(frames)
    if nf == n:
        return frames
    if nf < n:
        # try reg / n+1
        frames1 = select_by_DS3(l_reg / (n + 1))
        nf1 = len(frames1)
        if nf1 == n:
            return frames1
        if nf1 > n:
            _reg = l_reg / n
        else:
            _reg = l_reg / (n + 1)
            frames = frames1[:]
        _dir = -1
    else:
        _reg = l_reg / n
        _dir = 1

    # search from _dir
    _step = _dir * 0.01 * l_reg 
    if _dir < 0:
        while True:
            _reg +=  _step
            if _reg <= 0:
                break
            _frames = select_by_DS3(_reg)
            _nf = len(_frames)
            if _nf >= n:
                return _frames
            frames = _frames
    else:
        while True:
            _reg += _step
            if _reg > l_reg:
                break
            _frames = select_by_DS3(_reg)
            _nf = len(_frames)
            if _nf <= n:
                return _frames
            frames = _frames
    return frames

class FrameSelector(object):
    def __init__(self, prop_model):
        self.prop_model = prop_model
        self.dataname = self.prop_model.dataname
        self.features = prop_model.data.get_all_features()
        self.type = "ds3"    #"greedy"
        self.reg = 1
    
    def select_frames(self, vid, left, right, n):
        assert left <= right and n >= 1
        
        # get dist matrix
        video = []
        video.append(self.prop_model.data.get_idx_by_video_and_frame_id(vid, left))
        video.append(self.prop_model.data.get_idx_by_video_and_frame_id(vid, right))
        features = self.features[video[0]: video[1] + 1]
        dist = pairwise_distances(features, metric='euclidean')
        
        # subset selection
        if self.type == "greedy":
            subset_selector_solver = Greedy(dist, self.reg)
            frames, value = subset_selector_solver.randomized()
            frames.sort()

        if self.type == "ds3":
            frames = ds3_find_frames(dist, n)

        return frames

class LineFrameSelector(object):   
    def __init__(self, cdx, helper) -> None:
        self.class_name = cdx
        self.action_list = helper.action_classes[cdx]
        self.action_dict = {}
        for i, action in enumerate(self.action_list):
            self.action_dict[action['idx']] = i
    
    def select_frames(self, line: Line, subset, n):
        features = line.features[subset]
        dist = pairwise_distances(features, metric='euclidean')
        cols = ds3_find_frames(dist, n)
        ret = list(map(lambda x: int(subset[x]), cols))
        return ret
            


