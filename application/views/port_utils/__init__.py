from flask import jsonify
from .port import Port
import numpy as np
import os

port = Port()

def get_manifest():
    # manifest = {"image_num": 1}
    manifest = port.get_manifest()
    return jsonify(manifest)

def init_model(dataname, step):
    port.reset(dataname, step)
    return port

def get_hierarchy_meta_data():
    meta_data = port.model.get_hierarchy_meta_data()
    return jsonify(meta_data)

def get_hierarchy(h_id):
    port.run_model()
    # return jsonify({"test": 1})
    hierarchy = port.model.get_hierarchy(h_id)
    return jsonify(hierarchy)

def get_video_info(vid, action_class):
    scores = port.model.data.get_pred_score_of_video(vid, action_class)
    ret = {}
    ret["id"] = vid
    ret["length"] = len(scores)
    ret["scores"] = scores
    return jsonify(ret)

def get_rep_frames(vid, bound, target):
    frames = port.model.get_rep_frames(vid, bound, target)
    ret = {
        "id": vid,
        "N": len(frames),
        "frames": frames,
        "bound": bound
    }
    return jsonify(ret)

def get_single_frame(vid, fid):
    img = port.model.data.get_single_frame(vid, fid)
    return img

def get_video(vid):
    video = port.model.data.get_video(vid)
    return video

def get_icon(image_name):
    path = os.path.join(port.model.common_data_root, "Icons", image_name + ".png")
    return path

def get_alignment_of_anchor_action(cls, aid):
    res = port.model.get_alignment_of_anchor_action(cls, aid)
    return jsonify(res)

def get_pred_scores_of_video_with_given_boundary(vid, cls, start, end):
    bound, scores = port.model.get_pred_scores_of_video_with_given_boundary(vid, cls, start, end)
    ret = {
        "id": vid,
        "class": cls,
        "bound": bound,
        "scores": scores
    }
    return jsonify(ret)

def get_recommendation(class_name, action_id, labeled_frame_id, pos):
    state, result = port.model.recommend_by_prediction(class_name, action_id, labeled_frame_id, pos)
    ret = {
        "msg": "ok" if state else "ignore"
    }
    if ret["msg"] == "ok":
        ret.update(result)
    return ret

def update_by_constraint(constraint):
    result = port.model.update_constraints(constraint)
    ret = "ok" if result else "ignore"
    return jsonify({"msg": ret})

def remove_action(action_id):
    result = port.model.remove_action(action_id)
    ret = "ok" if result else "ignore"
    return jsonify({"msg": ret})

def get_history():
    result = port.model.prop_model.get_history_info()
    return jsonify(result)

def get_step_history(step, class_name):
    result = port.model.prop_model.get_history_action_length_by_class(step, class_name)
    return jsonify(result)