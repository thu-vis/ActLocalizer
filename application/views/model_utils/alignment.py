"""
This file include classes for alignements clustering and tree modify.
"""
import numpy as np
import time
from queue import Queue,PriorityQueue
from scipy.sparse import csr_matrix, hstack, vstack, find
# from .video_alignment import pairwise_video_alignment

class Line:
    """
    class for 1 alignment including size/width/features/frames/tree
    """
    def __init__(self, video, video_id, features):
        """
        init 1 video for 1 alignment

        Parameters:
        video - [x, y] frame index
        features -  features for video
        """
        # Alignment variables
        self.size = 1                                       # video nums in alignment
        self.index = video[0]                               # frame start index
        self.width = video[1] - video[0] + 1                # width of alignment
        self.features = features                            # (width, d) avg feature of frame
        self.frames = np.zeros((self.size, self.width), dtype='int32')     # (size, width) -1 means null    
        self.frames[0] = np.array(list(range(video[0], video[1] + 1)))
        self.ids = [video_id]                               # video_ids
        self.pair_numbers = np.zeros((self.width, ), dtype=np.int32)
        
        # Tree variables
        self.tree = {}                                          # tree structure
        self.tree['name'] = str(video_id)
        self.tree['frames'] = video
    
    def copy(self):
        # return a copy of alignment without tree
        ca = Line([0,0], 0, None)
        ca.size = self.size
        ca.index= self.index
        ca.width = self.width
        ca.features = np.copy(self.features)
        ca.frames = np.copy(self.frames)
        ca.ids = self.ids[:]
        ca.tree = {}
        return ca
        

    def merge(self, data, align1: list, align2: list, ac):
        """
        Merge another alignment into this alignment

        Parameters:
        data - another alignment
        align1/align2 - alignment of these 2 aligns
        ac - alignment object
        """
        assert self.features.shape[1] == data.features.shape[1]

        # 1. merge frames and features
        width = len(align1)
        features = np.zeros((width, self.features.shape[1]))

        frames_r1, frames_r2 = self.frames.T, data.frames.T
        frames1 = np.zeros((width, self.size))
        frames2 = np.zeros((width, data.size))

        i1, i2 = 0, 0
        for i in range(width):
            if align1[i] == -1:
                frames1[i] = -1 * np.ones(self.size)
            else:
                frames1[i] = frames_r1[i1]
                features[i] += self.size * self.features[i1]
                i1 += 1
            if align2[i] == -1:
                frames2[i] = -1 * np.ones(data.size)
            else:
                frames2[i] = frames_r2[i2]
                features[i] += data.size * data.features[i2]
                i2 += 1

        # 2. merge trees
        tree = {}
        tree['name'] = 'M' + str(ac.class_name) + '-' + str(ac.middle_index)
        ac.middle_index += 1
        tree['children'] = [self.tree, data.tree] 
        tree['aligns'] = [align1, align2]

        # 3. merge alignment meta info
        self.size += data.size
        self.ids += data.ids
        self.width = width
        self.features = features / self.size
        self.frames = np.c_[frames1, frames2].T
        self.tree = tree
        tree['ids'] = self.ids[:]

        # 4.merge index and affinity info
        index = ac.affinity_matrix.shape[0]
        new_frames = []
        i1, i2 = 0, 0
        for i in range(width):
            # new_frame = set()
            # if align1[i] != -1:
            #     new_frame = new_frame | ac.affinity_matrix[self.index + i1]
            #     i1 += 1
            # if align2[i] != -1:
            #     new_frame = new_frame | ac.affinity_matrix[data.index + i2]
            #     i2 += 1
            new_frame = dict()
            if align1[i] != -1:
                line_dict1 = ac.affinity_matrix[self.index + i1]
                for key in line_dict1.keys():
                    if key not in new_frame:
                        new_frame[key] = line_dict1[key]
                    else:
                        new_frame[key] += line_dict1[key]
                i1 += 1
            if align2[i] != -1:
                line_dict2 = ac.affinity_matrix[data.index + i2]
                for key in line_dict2.keys():
                    if key not in new_frame:
                        new_frame[key] = line_dict2[key]
                    else:
                        new_frame[key] += line_dict2[key]
                i2 += 1
            new_frames.append(new_frame)
        ac.affinity_matrix.append(new_frames)
        self.index = index


class Alignment:
    """
    class for Hierarchical Clustering for alignments
    """
    def __init__(self, class_name, actions, helper,
                    pre_aligned=False, alignments=None, pause=False):
        """
        init alignments and align matrix

        Parameters:
        n -  total number of videos in this action
        actions - list of actions
        helper - cluster helper
        """
        # clustering meta infomation
        self.class_name = class_name
        if pre_aligned:
            n = len(alignments)
        else:
            n = len(actions)
        self.pause = pause
        self.n_clusters = n                                    # current clusters
        self.minimum = -1000                                   # value of align_matrix[i,i]
        self.helper = helper
        self.alpha, self.beta = helper.alpha, helper.beta
        self.middle_index = 0                                  # current middle node number
        self.affinity_matrix = helper.affinity_matrix

        # alignments and align results
        if pre_aligned:
            self.alignments = self._clear_alignments(alignments)
        else:
            self.alignments = self._init_alignments(actions) 
        self.align_matrix = np.zeros((n, n))                   # dp result matrix (n, n)
        self.align_results = np.empty((n, n), dtype=list)      # align result D[i,j] = [id1, align1, align2] id1:an id in align1
        self._init_align_matrix()

        # print("Alignment Clustering: Cluster initialization succeeded.")

    def _init_alignments(self, actions):
        alignments, i = [], 0
        for action in actions:
            bound = self._get_action_bound(action)
            alignment = Line(bound, action['idx'], self.helper.features[bound[0]:bound[1]+1])
            alignments.append(alignment)
            i += 1
        return alignments

    def _get_action_bound(self, action):
        return [action['idx'] - action['single_frame'] + action['left_bound'], 
                action['idx'] - action['single_frame'] + action['right_bound']]

    def _clear_alignments(self, raw_alignments):
        alignments = []
        for alignment in raw_alignments:
            alignments.append(alignment.copy())
        return alignments

    def _init_align_matrix(self):
        """
        init align matrix by calculate (i,j)
        init M[i, i] = - Max
        """  
        for i in range(self.n_clusters):
            for j in range(i):
                align1, align2, self.align_matrix[j,i] = self.pairwise_align_alignment(
                    self.alignments[j], self.alignments[i])
                self.align_results[j, i]= [self.alignments[j].ids[0], align1, align2]
                self.align_results[i, j]= self.align_results[j, i]
                self.align_matrix[i, j] = self.align_matrix[j, i]
            self.align_matrix[i, i] = self.minimum
        return

    def pairwise_align_alignment(self, align1, align2, pause=False):
        """
        video alignment for 2 alignments

        Parameters:
            align1、align2 - alignment 1、2
            ac - current align clustering to calculate S and L1
                S - function to get S(a, b) [0,1]
                L1 - function to get L1(a, b) [0,1]

        Returns:
            aligned_video1 - [0,...,width1-1] list of index or -1 (means null)
            aligned_video2 - [0,...,width2-1] the same as video1
            F[l1][l2] - max alignment scores
        """
        # 1. initialize data and functions
        l1, l2 = align1.width, align2.width
        assert l1 > 0 and l2 > 0
        def T(i):   # get pos index of frame
            return i - 1

        def T1(i):  # get real index in all frames for align1
            return i + align1.index

        def T2(i):  # get real index in all frames for align1
            return i + align2.index

        def A(a, b): # find affinity matrix value
            if b in self.affinity_matrix[a]:
                return self.affinity_matrix[a][b]
            return 0

        def S(a, b): # value function for substitution
            return self.alpha * A(T1(a), T2(b)) + (1-self.alpha) * np.dot(align1.features[a], align2.features[b])
        
        def L1(a, b, mode): # value function for -
            return 0
            # if mode == 0:
            #     if b == -1:
            #         value = S(a, 0)
            #     else:
            #         value = S(0, b)
            # elif mode == 1 and b + 1 < align2.width:
            #     value = (S(a, b) + S(a, b+1)) / 2
            # elif mode == 2 and a + 1 < align1.width:
            #     value = (S(a, b) + S(a+1, b)) / 2
            # else:
            #     value = S(a, b)
            # return - self.beta * value

        # 2. dynamic program
        F = np.zeros((l1+1, l2+1))      # dp values
        C = np.zeros((l1+1, l2+1))      # choicea
        i, j = 2, 2
        C[0][0], C[1][0], C[0][1] = -1, 1, 2
        while i <= l1:
            F[i][0] = F[i-1][0] + L1(T(i), -1, 0)
            C[i][0] = 1
            i += 1
        while j <= l2:
            F[0][j] = F[0][j-1] + L1(-1, T(j), 0)
            C[0][j] = 2
            j += 1
        i, j = 1, 1
        while i <= l1:
            while j <= l2:
                values = [F[i-1][j-1] + S(T(i),T(j)),
                            F[i-1][j] + L1(T(i), T(j), 1),
                            F[i][j-1] + L1(T(i), T(j), 2)]
                choice = np.argmax(np.array(values, dtype=np.float64))
                F[i][j] = values[choice]
                C[i][j] = choice
                j += 1
            j = 1
            i += 1

        if self.pause and pause:
            from IPython import embed
            embed()

        # Backtrack and generate path
        stack1, stack2 = [] , []
        ptr1, ptr2 = l1, l2
        choice = C[l1][l2]
        while choice >= 0:
            if choice == 0:
                stack1.append(T(ptr1))
                stack2.append(T(ptr2))
                ptr1 -= 1
                ptr2 -= 1
            elif choice == 1:
                stack1.append(T(ptr1))
                stack2.append(-1)
                ptr1 -= 1
            else:
                stack1.append(-1)
                stack2.append(T(ptr2))
                ptr2 -= 1
            choice = C[ptr1][ptr2]
        
        aligned_video1 = stack1[::-1]
        aligned_video2 = stack2[::-1]
        return aligned_video1, aligned_video2, F[l1][l2]
        
    def cluster_step(self):
        """
        1 step for clustering: n to n-1
        """
        # 1. find max pos [TODO: can be faster]
        assert self.n_clusters > 1
        i, j = np.unravel_index(self.align_matrix.argmax(), self.align_matrix.shape)
        if i > j:
            i, j = j, i
        # if self.pause:
        #     from IPython import embed
        #     embed()
        # self.pairwise_align_alignment(
        #             self.alignments[i], self.alignments[j], pause=True)

        # 2. merge i and j into i
        temp_id, align1, align2 = self.align_results[i, j]
        if temp_id not in self.alignments[i].ids:
            align1, align2 = align2, align1
        self.alignments[i].merge(self.alignments[j], align1, align2, self)

        # 3. swap j and n-1
        final = self.n_clusters-1
        self.alignments[j] = self.alignments[final]
        self.alignments[final] = None

        self.align_matrix[j], self.align_matrix[final] = self.align_matrix[final], self.align_matrix[j]
        self.align_matrix = self.align_matrix.T
        self.align_matrix[j], self.align_matrix[final] = self.align_matrix[final], self.align_matrix[j]

        self.align_results[j], self.align_results[final] = self.align_results[final], self.align_results[j]
        self.align_results = self.align_results.T
        self.align_results[j], self.align_results[final] = self.align_results[final], self.align_results[j]

        # 4. delete n-1
        self.alignments.pop()
        self.align_matrix = self.align_matrix[:-1,:-1]
        self.align_results = self.align_results[:-1, :-1]
        self.n_clusters -= 1

        # 5. recalculate matrix
        for j in range(self.n_clusters):
            if i == j:
                continue
            align1, align2, self.align_matrix[j, i] = self.pairwise_align_alignment(
                    self.alignments[j], self.alignments[i])
            self.align_matrix[i, j] = self.align_matrix[j, i]
            self.align_results[j, i] = [self.alignments[j].ids[0], align1, align2]
            self.align_results[i, j] = self.align_results[j, i]
            
    def cluster_to(self, n):
        assert n <= self.n_clusters
        while self.n_clusters > n:
            self.cluster_step()
            # if self.n_clusters % 5 == 0:
            #     print("Alignment Clustering: cluster to ", self.n_clusters, " clusters.")
        
        #import IPython; IPython.embed(); exit()


        
