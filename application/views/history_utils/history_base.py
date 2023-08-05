import numpy as np
import os
from time import time


class HistoryBase(object):
    def __init__(self, dataname=None):
        # set dataname
        self.dataname = dataname
        
        # data-state-related variables
        self.state_idx = 0
        self.state = {}
        self.state_data = {}
        self.current_state = None

    def set_dataname(self, dataname):
        self.dataname = dataname

    def record_state(self, pred):
        new_state = Node(self.state_idx, parent=self.current_state)
        self.state_idx = self.state_idx + 1
        self.current_state = new_state
        self.state_data[self.current_state.name] = {
            "test": 1
        }
        self.print_state()

    # this function is for DEBUG
    def print_state(self):
        dict_exporter = DictExporter()
        tree = dict_exporter.export(self.state)
        print(tree)
        print("current state:", self.current_state.name)

    def return_state(self):
        return {
            "test": 1,
            # "current_id": int(self.current_state.name)
            "current_id": int(1)
        }

    def change_state(self, id):
        state = self.state_data[id]["state"]
        self.current_state = state
        data = self.state_data[self.current_state.name]
        self.affinity_matrix = data["affinity_matrix"]
        self.train_idx = data["train_idx"]
        self.selected_labeled_idx = data["selected_labeled_idx"]
        self.print_state()
        return self.return_state()