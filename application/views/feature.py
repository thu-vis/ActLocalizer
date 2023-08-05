from flask import Blueprint, render_template, abort, session, send_file, request, url_for

from application.views.database_utils.data import Data
from application.views.features.feature import VideoFeature

feature = Blueprint("feature", __name__)
# dt = Data("thumos19", 0)
# vf = VideoFeature(dt)

# prefix = "/feature"
@feature.route("/data/")
def get_tsne_data():
    result, label = vf.read_tsne()
    if type(label) != list:
        label = label.tolist()
    return {
        "msg": "ok",
        "result": result.tolist(),
        "label": label,
        "length": len(label)
    }

@feature.route("/info/")
def get_feature_info():
    return {
        "msg": "ok",
        "cls": vf.cls,
        "labels": vf.data.meta_data["classes"],
        "map": vf.data.meta_data["class_2_id_map"]
    }

@feature.route("/test/")
def test_video():
    vf.extract_video_action_features(114)
    return {

    }