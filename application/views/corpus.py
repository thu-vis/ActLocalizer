from flask import Blueprint, render_template, abort, session, send_file, request, url_for
import json
import time

from .database_utils.data import Data
from .features.feature import VideoFeature

from .port_utils import *

corpus = Blueprint("corpus", __name__)

@corpus.route("/GetManifest", methods=["GET", "POST"])
def app_get_manifest():
    # extract info from request
    dataname = json.loads(request.data)["dataset"]
    step = int(json.loads(request.data)["step"])
    init_model(dataname, step)
    return get_manifest()

@corpus.route("/Update", methods=["GET", "POST"])
def app_update():
    data = json.loads(request.data)
    constraint = data["constraint"]
    return update_by_constraint(constraint)

@corpus.route("/RemoveAction", methods=["GET", "POST"])
def app_remove_action():
    data = json.loads(request.data)
    action_id = data["action_id"]
    return remove_action(action_id) 

@corpus.route("/GetHiearchyMetaData", methods=["GET", "POST"])
def app_get_hierarchy_meta_data():
    return get_hierarchy_meta_data()

@corpus.route("/GetHierarchy", methods=["GET", "POST"])
def app_get_hierarchy():
    data = json.loads(request.data)
    h_id = data["id"]
    return get_hierarchy(h_id)

@corpus.route("/ActionIcon", methods=["GET"])
def app_get_icon():
    image_name = request.args["filename"]
    path = get_icon(image_name)
    return send_file(path)

@corpus.route("/SetCorpusArgs", methods=["GET", "POST"])
def app_set_corpus_args():
    data = json.loads(request.data)
    arg_id = data["arg_id"]
    value = data["value"]
    if arg_id == 0:
        port.model.nearest_actions = value
    elif arg_id == 1:
        port.model.nearest_frames = value
    elif arg_id == 2:
        port.model.window_size = value
    return jsonify({"msg": "ok"})


@corpus.route("/GetRecommendation", methods=["GET", "POST"])
def app_get_recommendation():
    data = json.loads(request.data)
    # action_id: vid-fid
    # labeled_frame_id: vid-fid
    class_name = data["class"]
    action_id = data["action_id"]
    labeled_frame_id = data["labeled_frame_id"]
    bound_pos = data["bound_pos"]
    return get_recommendation(class_name, action_id, labeled_frame_id, bound_pos)


@corpus.route("/GetHistory", methods=["GET", "POST"])
def app_get_history():
    return get_history()

@corpus.route("/GetStepHistory", methods=["GET", "POST"])
def app_get_step_history():
    data = json.loads(request.data)
    step = data["step"]
    class_name = data["class"]
    return get_step_history(step, class_name)