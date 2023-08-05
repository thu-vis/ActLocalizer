from multiprocessing.dummy import active_children
from flask import Blueprint, render_template, abort, session, send_file, request, url_for, Response
import json
import time

from .database_utils.data import Data
from .features.feature import VideoFeature

from .port_utils import *

video = Blueprint("video", __name__)

video_file = None

@video.route("/GetVideo", methods=["GET"])
def app_get_video():
    global video_file
    vid = int(request.args["id"])
    video_path = get_video(vid)
    print("video_path", video_path)
    video_size = os.path.getsize(video_path)
    if video_file != None:
        video_file.close()
        video_file = None
    video_file = open(video_path, "rb")
    video_file.seek(request.range.ranges[0][0])
    headers = {
        'Accept-Range': 'bytes',
        'Content-Length': video_size,
        'Content-Range': request.range.to_content_range_header(video_size)
    }
    return Response(video_file, 206, headers, content_type='video/mp4')  # send_file(video_path)

@video.route("/GetVideoInfo", methods=["GET", "POST"])
def app_get_video_info():
    data = json.loads(request.data)
    vid = data["id"]
    action_class = data["class"]
    return get_video_info(vid, action_class)

@video.route("/GetRepFrames", methods=["GET", "POST"])
def app_get_rep_frames():
    data = json.loads(request.data)
    vid = data["id"]
    bound = data["bound"]
    target = data["target"]
    return get_rep_frames(vid, bound, target)

@video.route("/GetAlignmentOfAnchorAction", methods=["GET", "POST"])
def app_get_alignment_of_anchor_action():
    data = json.loads(request.data)
    action_class = data["class"]
    aid = data["id"]
    # k = data["k"] # k = -1 means using the default k
    return get_alignment_of_anchor_action(action_class, aid)

@video.route("/GetPredScoresOfVideoBoundary", methods=["GET", "POST"])
def app_get_pred_scores_of_video_boundary():
    data = json.loads(request.data)
    vid = data["id"]
    action_class = data["class"]
    start, end = data["bound"]
    return get_pred_scores_of_video_with_given_boundary(vid, action_class, start, end)
