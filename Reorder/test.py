import numpy as np
from application.views.utils.helper_utils import pickle_load_data
from Reorder import reorder

def get_dists(data, id):
    """
    Get distance matrix(n * n) of the cluster including given action 
    params:
        data - data of info
        id - action id (class 3: [0,39])
    """
    m_index = data['dict'][id]
    dists = data['dists'][m_index]
    return dists.sum(axis=2)

def get_cross(data, id):
    """
    Get cross matrix(n * n * w) of switching frame when selecting the given action as a reference
    params:
        data - data of info
        id - action id (class 3: [0,39])
    """
    m_index = data['dict'][id]
    cross = data['cross'][m_index][id]
    return cross

def get_switch(data, id):
    """
    Get switch matrix(n * n * w) of switching frame when selecting the given action as a reference
    params:
        data - data of info
        id - action id (class 3: [0,39])
    """
    m_index = data['dict'][id]
    switch = data['switch'][m_index][id]
    return switch

def get_cross_switch(data, id, alpha):
    return get_cross(data, id) + alpha * get_switch(data, id) 

if __name__ == "__main__":
    class_id = 3
    data = pickle_load_data("data_class_{}.pkl".format(class_id))
    action_id = 0
    alpha = 2

    dists = get_dists(data, action_id)
    cross_switch = get_cross_switch(data, action_id, alpha)