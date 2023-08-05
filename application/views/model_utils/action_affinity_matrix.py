
class ActionAffinityMatrix(object):
    def __init__(self, cdx, frame_set, all_affinity_matrix):
        """
        Params:
            cdx - class index
            frame_set - action frame set of class 
            all_affinity_matrix - affinity matrix of all frames
        """
        self.class_name = cdx
        self.values = {}
        self.shape = all_affinity_matrix.shape
        for frame in frame_set:
            value_set = set(all_affinity_matrix[frame].nonzero()[1]) & frame_set
            self.values[frame] = dict.fromkeys(value_set, 1)
    
    def __getitem__(self, key):
        # try:
        #     assert key in self.values
        # except:
        #     from IPython import embed
        #     embed();exit()
        return self.values[key]

    # def append(self, sets):
    #     index = self.shape[0]
    #     width = len(sets)
    #     for i in range(width):
    #         for value in sets[i]:
    #             self.values[value].add(index + i)
    #         self.values[index + i] = sets[i]
    #     self.shape = (index + width, index + width)

    def append(self, dicts):
        index = self.shape[0]
        width = len(dicts)
        for i in range(width):
            for value in dicts[i].keys():
                self.values[value][index + i] = dicts[i][value]
            self.values[index + i] = dicts[i]
        self.shape = (index + width, index + width)
        
    
    # def append_i(self, sets, index):
    #     assert index not in self.values 
    #     width = len(sets)
    #     for i in range(width):
    #         for value in sets[i]:
    #             self.values[value].add(index + i)
    #         self.values[index + i] = sets[i]
    #     shape = max(self.shape[0], index + width)
    #     self.shape = (shape, shape)

    def append_i(self, dicts, index):
        assert index not in self.values
        index = self.shape[0]
        width = len(dicts)
        for i in range(width):
            for value in dicts[i].keys():
                self.values[value][index + i] = dicts[i][value]
            self.values[index + i] = dicts[i]
        shape = max(self.shape[0], index + width)
        self.shape = (shape, shape)