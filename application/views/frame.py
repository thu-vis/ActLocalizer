from ipaddress import ip_address
from flask import Blueprint, render_template, abort, session, send_file, request, url_for
import json
import time
from io import BytesIO
from PIL import Image

from .database_utils.data import Data
from .features.feature import VideoFeature

from .port_utils import *

frame = Blueprint("frame", __name__)

def serve_pil_image(pil_img):
    pil_img = Image.fromarray(pil_img)
    size = max(pil_img.size)
    pil_img = pil_img.resize([size, size])
    print("size", pil_img.size)
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality="web_high")
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@frame.route("/single_frame", methods=["GET"])
def app_get_single_frame():
    vid, fid = request.args["filename"].split("-")
    vid = int(vid)
    fid = int(fid)
    frame_path = get_single_frame(vid, fid)
    print("frame_path", frame_path)
    return send_file(frame_path)
    # img = get_single_frame(vid, fid)
    # img = img[:, :, ::-1]
    # import IPython; IPython.embed(); exit()
    # return serve_pil_image(img)
    # img_path = get_origin_image(idx)
    # return send_file(img_path)