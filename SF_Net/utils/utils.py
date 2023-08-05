from shutil import register_archive_format
import numpy as np
import pickle
import json
import os
import time
import ctypes
import logging
import cv2
try:
    import torch
except:
    None
    
def common_data(list1, list2):  
    result = False  
    for x in list1:  
        for y in list2:  
            if x == y:  
                result = True
                return result
    return result

def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i].decode('utf-8')][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)], axis=0)


def strlist2onehot(strlist, classlist):
    return np.eye(len(classlist))[strlist2indlist(strlist, classlist)]


def idx2multihot(id_list, num_class):
    return np.sum(np.eye(num_class)[id_list], axis=0)


def random_extract(feat, t_max):
    r = np.random.randint(len(feat)-t_max)
    return feat[r:r+t_max]


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
        return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length):
    if len(feat) > length:
        return random_extract(feat, length)
    else:
        return pad(feat, length)


def write_to_file(dname, dmap, cmap, itr):
    fid = open(dname + '-results.log', 'a+')
    string_to_write = str(itr)
    for item in dmap:
        string_to_write += ' ' + '%.2f' % item
    string_to_write += ' ' + '%.2f' % cmap
    #string_to_write += ' ' + '%.2f' %cmap1
    fid.write(string_to_write + '\n')
    fid.close()

def feature_norm(features):
    norms = (features**2).sum(axis=1)
    norms = norms ** 0.5
    features = features / norms[:, np.newaxis]
    return features

# Pickle loading and saving
def pickle_save_data(filename, data):
    try:
        pickle.dump(data, open(filename, "wb"))
    except Exception as e:
        print(e, end=" ")
        print("So we use the highest protocol.")
        pickle.dump(data, open(filename, "wb"), protocol=4)
    return True


def pickle_load_data(filename):
    try:
        mat = pickle.load(open(filename, "rb"))
    except Exception:
        mat = pickle.load(open(filename, "rb"))
    return mat


# json loading and saving
def json_save_data(filename, data):
    with open(filename, "w") as f:
        f.write(json.dumps(data))
    return True


def json_load_data(filename, encoding=None):
    with open(filename, "r", encoding=encoding) as f:
        return json.load(f)


# directory
def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return True


def check_exist(filename):
    return os.path.exists(filename)


# model saver and loader
def save_checkpoint(iteration, net, optimizer, optimizer_centloss_f,\
    optimizer_centloss_r, checkpoint_file, is_best=False):
    logger = get_logger()
    check_dir(checkpoint_file)
    checkpoint_log = os.path.join(checkpoint_file, "checkpoint")
    if os.path.exists(checkpoint_log):
        d = json_load_data(checkpoint_log)
    else:
        d = {}
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_centloss_f_state_dict": optimizer_centloss_f.state_dict(),
        "optimizer_centloss_r_state_dict": optimizer_centloss_r.state_dict(),
    }
    torch.save(checkpoint, os.path.join(checkpoint_file, "sfnet.{}.pkl".format(iteration)))
    logger.info("save models in {}".format(os.path.join(checkpoint_file, "sfnet.{}.pkl".format(iteration))))
    d = {
        "last_model": "sfnet.{}.pkl".format(iteration)
    }
    if is_best:
        d["best_model"] = d["last_model"]
    json_save_data(checkpoint_log, d)


def resume_checkpoint(resume, checkpoint_file, eval_debug=False):
    logger = get_logger()
    # if isinstance(resume, bool):
    # if eval_debug is True:
    #     checkpoint = torch.load(resume)
    #     return 1000, checkpoint, False, False, False

    if resume in ["True", "False"]:
        resume = bool(resume)
        if resume:
            checkpoint_log = os.path.join(checkpoint_file, "checkpoint")
            if not os.path.exists(checkpoint_log):
                return 0, False, False, False, False
            d = json_load_data(checkpoint_log)
            last_model = d["last_model"]
            checkpoint = torch.load(os.path.join(checkpoint_file, last_model))
            logger.info("*******************************************************************************")
            logger.info("*******************************************************************************")
            logger.info("load model from {}".format(os.path.join(checkpoint_file, last_model)))
            logger.info("*******************************************************************************")
            logger.info("*******************************************************************************")
            return checkpoint["iteration"], checkpoint["model_state_dict"], \
                checkpoint["optimizer_state_dict"], \
                checkpoint["optimizer_centloss_f_state_dict"], \
                checkpoint["optimizer_centloss_r_state_dict"]

        else:
            return 0, False, False, False, False

    elif isinstance(resume, str):
        logger.info("*******************************************************************************")
        logger.info("*******************************************************************************")
        logger.info("load model from {}".format(resume))
        logger.info("*******************************************************************************")
        logger.info("*******************************************************************************")
        checkpoint = torch.load(resume)
        return checkpoint["iteration"], checkpoint["model_state_dict"], \
                checkpoint["optimizer_state_dict"], \
                checkpoint["optimizer_centloss_f_state_dict"], \
                checkpoint["optimizer_centloss_r_state_dict"]
    elif resume is None:
        return 0, False, False, False, False


# video utils
def read_single_frame(v_filename, frame_idx, stride, fps, original_fps):
    cap = cv2.VideoCapture(v_filename)
    f_id = frame_idx * stride + stride / 2
    f_id = int(f_id / fps * original_fps)
    print("f_id", f_id, "frame_idx", frame_idx)
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_id)
    rval, frame = cap.read()
    return frame

def save_frames(v_filename, frame_idxs, stride, fps, original_fps, out_dir):
    v_name = os.path.split(v_filename)[1]
    v_name = v_name.split(".")[0]
    for frame_idx in frame_idxs:
        frame = read_single_frame(v_filename, frame_idx, stride, fps, original_fps)
        outfilename = os.path.join(out_dir, v_name + "-" + str(frame_idx) + ".jpg")
        cv2.imencode(".jpg", frame)[1].tofile(outfilename)
    

# Logger 
def strftime(t=None):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t or time.time()))


logging.basicConfig(format="[ %(asctime)s][%(module)s.%(funcName)s] %(message)s")

DEFAULT_LEVEL = logging.INFO
# DEFAULT_LOGGING_DIR = "log"
# if not os.path.exists(DEFAULT_LOGGING_DIR):
#     os.makedirs(DEFAULT_LOGGING_DIR)
fh = None

def init_fh(log_dir):
    global fh
    if fh is not None:
        return
    if log_dir is None:
        raise ValueError("log_dir should be given")
    # if DEFAULT_LOGGING_DIR is None:
    #     return
    # if not os.path.exists(DEFAULT_LOGGING_DIR): os.makedirs(DEFAULT_LOGGING_DIR)
    # t = strftime()
    # day_path = os.path.join(DEFAULT_LOGGING_DIR, t.split("-")[0])
    # check_dir(day_path)
    logging_path = os.path.join(log_dir, strftime() + ".log")
    fh = logging.FileHandler(logging_path)
    fh.setFormatter(logging.Formatter("[ %(asctime)s][%(module)s.%(funcName)s] %(message)s"))

def update_default_level(defalut_level):
    global DEFAULT_LEVEL
    DEFAULT_LEVEL = defalut_level

def update_default_logging_dir(default_logging_dir):
    global DEFAULT_LOGGING_DIR
    DEFAULT_LOGGING_DIR = default_logging_dir

def get_logger(log_dir=None, name="FS", level=None):
    level = level or DEFAULT_LEVEL
    logger = logging.getLogger(name)
    logger.setLevel(level)
    init_fh(log_dir)
    if fh is not None:
        logger.addHandler(fh)
    return logger

# logger = get_logger()
logger = None

# def init_logger(log_dir):
#     global logger
#     logger = get_logger(log_dir)