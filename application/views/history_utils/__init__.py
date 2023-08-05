from .history_base import HistoryBase

history_base = HistoryBase()

def history_set_dataname(dataname):
    history_base.set_dataname(dataname)

def get_history():
    # return {"test": "get_history"}
    history_data = history_base.return_state()

    # for test
    history_data = []
    history_data.append({
        "id": 0,
        "children": [1]
    })
    history_data.append({
        "id": 1,
        "children": [2, 3]
    })
    history_data.append({
        "id": 2,
        "children": []
    })
    history_data.append({
        "id": 3,
        "children": []
    })

    return history_data

def set_history(id):
    # return {"test": "set_history"}
    state = history_base.change_state(id)
    return 0

def retrain():
    return {"test": "retrain"}
