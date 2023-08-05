# import utils
import numpy as np
import time
from scipy.spatial import distance_matrix as dm
from copy import deepcopy
from ..utils import get_logger, check_dir
from .propagation import propagation_base_on_graph
from ..kNN_utils import get_features, feature_norm
from ..kNN_utils import left_bound, right_bound
import os
try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from tensorboard_logger import log_value
    from torch.autograd import Variable
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
except:
    None

def video_loss(logits, seq_len, labels, device):
    ''' 
        logits: torch tensor of dimension (B, n_element, n_class),
        seq_len: numpy array of dimension (B,) indicating the length of each video in the batch, 
        labels: torch tensor of dimension (B, n_class) of 1 or 0
        return: torch tensor of dimension 0 (value) 
    '''

    k = np.ceil(seq_len / 8).astype('int32')
    labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-10)
    lab = torch.zeros(0).to(device)
    instance_logits = torch.zeros(0).to(device)
    for i in range(len(logits)):
        if seq_len[i] < 5 or labels[i].sum() == 0:
            continue
        tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
        lab = torch.cat([lab, labels[[i]]], dim=0)
    clsloss = -torch.mean(torch.sum(
        Variable(lab) * F.log_softmax(instance_logits, dim=1), dim=1),
        dim=0)
    return clsloss

def center_loss(features, logits, labels, seq_len, criterion, itr, device):
    ''' 
        features: torch tensor dimension (B, n_element, feature_size),
        logits: torch tensor of dimension (B, n_element, n_class),
        labels: torch tensor of dimension (B, n_class) of 1 or 0,
        seq_len: numpy array of dimension (B,) indicating the length of each video in the batch, 
        criterion: center loss criterion, 
        return: torch tensor of dimension 0 (value)
    '''

    lab = torch.zeros(0).to(device)
    feat = torch.zeros(0).to(device)
    itr_th = 500
    for i in range(features.size(0)):
        if (labels[i] > 0).sum() == 0 or ((labels[i] > 0).sum() != 1
                                          and itr < itr_th):
            continue
        # categories present in the video
        labi = torch.arange(labels.size(1))[labels[i] > 0]
        atn = F.softmax(logits[i][:seq_len[i]], dim=0)
        atni = atn[:, labi]
        # aggregate features category-wise
        for l in range(len(labi)):
            labl = labi[[l]].float()
            atnl = atni[:, [l]]
            atnl[atnl < atnl.mean()] = 0
            sum_atn = atnl.sum()
            if sum_atn > 0:
                atnl = atnl.expand(seq_len[i], features.size(2))
                # attention-weighted feature aggregation
                featl = torch.sum(features[i][:seq_len[i]] * atnl,
                                  dim=0,
                                  keepdim=True) / sum_atn
                feat = torch.cat([feat, featl], dim=0)
                lab = torch.cat([lab, labl], dim=0)

    if feat.numel() > 0:
        # Compute loss
        loss = criterion(feat, lab)
        return loss / feat.size(0)
    else:
        return 0

def frame_loss(logits, frame_ids, seq_len, act_labels, device, background=False, tm=1):
    ''' 
        logits: torch tensor of dimension (B, n_element, n_class),
        seq_len: numpy array of dimension (B,) indicating the length of each video in the batch, 
        mid_ids: numpy array of dimesnion (B, max_seq)
        act_labels: torch tensor of dimension (B, n_class) of 1 or 0
        return: torch tensor of dimension 0 (value) 
    '''

    act_logits = torch.cat(
        [logits[i][frame_ids[i]] for i in range(len(logits))], dim=0)
    clsloss = -torch.mean(torch.sum(
        Variable(act_labels) * F.log_softmax(act_logits, dim=1), dim=1),
        dim=0)
    if background:
        bg_logits = []
        bg_count = 0
        for i in range(len(logits)):
            k = min(int(len(frame_ids[i]) * tm),
                    int(seq_len[i]) - len(frame_ids[i]))
            if k < 1:
                continue
            bg_count += k
            no_lab_id = list(set(range(seq_len[i])) - set(frame_ids[i]))
            bg_logits += [logits[i][no_lab_id]]
        if bg_count < 1:
            return clsloss
        bg_logits = torch.cat(bg_logits, dim=0)
        _, inds = torch.topk(bg_logits, k=bg_count, dim=0)
        bg_logits = bg_logits[inds[:, 0]]
        lab = np.zeros((bg_count, bg_logits.size(-1)))
        lab[:, 0] = 1.0
        labels = torch.from_numpy(lab).float().to(device)
        bgloss = -torch.mean(torch.sum(
            Variable(labels) * F.log_softmax(bg_logits, dim=1), dim=1),
            dim=0)
        clsloss += bgloss / logits.size(-1)
    return clsloss

def act_loss(logits, frame_ids, seq_len, device, tm=1):
    ''' 
        logits: torch tensor of dimension (B),
        seq_len: numpy array of dimension (B) indicating the length of each video in the batch, 
        mid_ids: numpy array of dimesnion (B, max_seq)
        return: torch tensor of dimension 0 (value) 
    '''

    instance_logits = torch.cat(
        [logits[i][frame_ids[i]] for i in range(len(logits))], dim=0)
    clsloss = -torch.mean(F.logsigmoid(instance_logits))
    bg_logits = []
    bg_count = 0
    for i in range(logits.size(0)):
        k = min(int(len(frame_ids[i]) * tm),
                int(seq_len[i]) - len(frame_ids[i]))
        if k < 1:
            continue
        bg_count += k
        no_lab_id = list(set(range(seq_len[i])) - set(frame_ids[i]))
        bg_logits += [logits[i][no_lab_id]]
    if bg_count < 1:
        return clsloss
    bg_logits = torch.cat(bg_logits, dim=0)
    bg_logits, _ = torch.sort(bg_logits)
    bg_logits = bg_logits[-bg_count:]
    clsloss = clsloss - torch.mean(torch.log(1+1e-3-torch.sigmoid(bg_logits)))
    return clsloss

def train_SF(itr,
             dataset,
             args,
             model,
             optimizer,
             criterion_cent_all,
             optimizer_centloss_all,
             tb_writer,
             device,
             ce,
             params,
             mode='single'):

    criterion_cent_f = criterion_cent_all[0]
    criterion_cent_r = criterion_cent_all[1]
    optimizer_centloss_f = optimizer_centloss_all[0]
    optimizer_centloss_r = optimizer_centloss_all[1]
    centloss_itr = 0
    total_loss = 0
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    logger = get_logger()

    # Batch fprop
    features, labels, count_labels, frame_labels, frame_ids = dataset.load_frame_data(
    )
    if args.background:
        labels = np.pad(labels, ((0, 0), (1, 0)), mode='constant')
        count_labels = np.pad(count_labels, ((0, 0), (1, 0)), mode='constant')
        frame_labels = np.pad(frame_labels, ((0, 0), (1, 0)), mode='constant')
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    count_labels = torch.from_numpy(count_labels).float().to(device)

    # model
    features_f, logits_f, features_r, logits_r, tcam, att_logits_f, att_logits_r, att_logits = model(
        Variable(features), device, seq_len=torch.from_numpy(seq_len).to(device))

    vloss_f = video_loss(logits_f, seq_len, labels, device)
    vloss_r = video_loss(logits_r, seq_len, labels, device)
    vloss_final = video_loss(tcam, seq_len, labels, device)
    vloss = vloss_f + vloss_r + vloss_final
    tb_writer.log_value('loss/video_loss', vloss, itr)
    total_loss += vloss * alpha

    if mode == 'weakly':
        centloss_f = center_loss(features_f, logits_f, labels, seq_len,
                                 criterion_cent_f, itr,
                                 device) * gamma
        optimizer_centloss_f.zero_grad()
        # center loss
        centloss_r = center_loss(features_r, logits_r, labels, seq_len,
                                 criterion_cent_r, itr,
                                 device) * gamma
        optimizer_centloss_r.zero_grad()
        centloss = centloss_f + centloss_r
        total_loss += centloss
    else:
        flabels = torch.from_numpy(frame_labels).float().to(device)
        floss_f = frame_loss(logits_f, frame_ids, seq_len, flabels,
                             device, background=args.background, tm=args.tm)
        floss_r = frame_loss(logits_r, frame_ids, seq_len, flabels,
                             device, background=args.background, tm=args.tm)
        floss_final = frame_loss(
            tcam, frame_ids, seq_len, flabels, device, background=args.background, tm=args.tm)
        floss = floss_f + floss_r + floss_final
        tb_writer.log_value('loss/frame_loss', floss, itr)
        total_loss += floss

        aloss_f = act_loss(att_logits_f, frame_ids,
                           seq_len, device, tm=args.tm)
        aloss_r = act_loss(att_logits_r, frame_ids,
                           seq_len, device, tm=args.tm)
        aloss_final = act_loss(
            att_logits, frame_ids, seq_len, device, tm=args.tm)
        aloss = aloss_f + aloss_r + aloss_final
        tb_writer.log_value('loss/act_loss', aloss, itr)
        if aloss > 0.1:
            total_loss += aloss * beta
        tb_writer.log_value('loss/total_loss', total_loss, itr)

    # logger.info('Iteration: %d, Loss: %.3f ' % (itr, total_loss))

    tb_writer.log_value('total_loss', total_loss, itr)

    optimizer.zero_grad()
    if total_loss.item() > 0:
        total_loss.backward()
    # Update centers
    if itr > centloss_itr:
        for param in criterion_cent_f.parameters():
            if param.grad is not None:
                param.grad.data *= (1. / beta)
        optimizer_centloss_f.step()
        for param in criterion_cent_r.parameters():
            if param.grad is not None:
                param.grad.data *= (1. / beta)
        optimizer_centloss_r.step()
    # Update model params
    if total_loss.item() > 0:
        optimizer.step()

def anchor_expand(logits, label, centers, radious=3, pv=0.5):
    frame_label = deepcopy(label)
    cls_scores = logits
    anchor_frames = [
        i for i in range(len(frame_label)) if len(frame_label[i]) > 0
    ]
    vlength = len(cls_scores)
    anchors = []
    for i in range(len(anchor_frames)):
        idx = anchor_frames[i]
        anchor_label = frame_label[idx]
        pa_label = np.argmax(cls_scores[idx])
        anchor_cls_score = np.mean(cls_scores[idx][anchor_label])
        def _expand(v):
            step = 0
            while True:
            # for step in range(radious):
                s_idx = idx + v
                cur_idx = idx + v*(step+1)
                e_idx = idx + v*(step+2)
                # the following 4 lines ensure that the idxs are within the video range
                min_idx = np.min([s_idx, cur_idx, e_idx])
                max_idx = np.max([s_idx, cur_idx, e_idx])
                if min_idx < 0 or max_idx >= vlength:
                    break
                # the following 2 lines ensure the expanding stops when meet another labeled frame
                if len(frame_label[cur_idx]) > 0:
                    break
                score = np.mean(cls_scores[cur_idx][anchor_label])
                ps_label = np.argmax(cls_scores[s_idx])
                pc_label = np.argmax(cls_scores[cur_idx])
                pe_label = np.argmax(cls_scores[e_idx])
                if ps_label == pc_label and pc_label == pe_label:
                    if score >= anchor_cls_score * pv:
                        frame_label[cur_idx] = frame_label[idx]
                    else:
                        break
                else:
                    break
                step += 1
        _expand(-1)
        _expand(1)
    return frame_label

def act_expand(args, dataset, model, device, radious=3, pv=0.95, centers=None, prop=False):
    logger = get_logger()
    classlist = dataset.get_classlist()
    right = np.zeros(len(classlist))
    count = np.zeros(len(classlist))
    gt_count = np.zeros(len(classlist))
    frame_level_count = 0
    frame_level_right = 0
    # Batch fprop
    train_idx = dataset.get_trainidx() # train idx指的是哪些数据？
    expand_count = 0
    classlist = dataset.get_classlist()
    centers = [[] for _ in range(len(classlist))]
    # outputs = []
    outputs = {}
    for idx in train_idx:
        feat = dataset.get_feature(idx)
        feat = torch.from_numpy(np.expand_dims(feat,
                                               axis=0)).float().to(device)
        cur_label = dataset.get_single_label(idx) # cur_label is single frame labels
        with torch.no_grad():
            _, logits_f, _, logits_r, tcam, _, _, _ = model(
                Variable(feat), device, is_training=False)
            tcam = tcam.data.cpu().numpy().squeeze()
            # if args.background: # 为True的话，把背景帧去掉了？？
            #     tcam = tcam[:, 1:]
            assert len(cur_label) == len(tcam)
            for jdx, ls in enumerate(cur_label):
                if len(ls) > 0:
                    for l in ls:
                        centers[l].append(tcam[jdx]) # centers[l]中存的是标注为l的所有样本
                # if dataset.is_actively_labeled(idx, jdx): #这句是啥意思
                #     cur_label[jdx] = []
            # outputs += [[idx, cur_label, tcam]]
            outputs[idx] = [idx, cur_label, tcam] # idx: 
    if prop:
        alpha_1 = 0.5
        alpha_2 = 1.0
        beta = 1.0
        pv = 0.90
        propagation_base_on_graph(dataset, outputs, args.A, args.buffer_path, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta)
    if args.export:
        export_dir = os.path.join(args.dir, "export_data")
        check_dir(export_dir)
        logits = []
        for output in outputs:
            logit = outputs[output][2]
            logits.append(logit)
        logits = np.concatenate(logits)
        np.save(os.path.join(export_dir, "propagation.npy"), logits)
    for output in outputs:
        idx = output
        cur_label = outputs[output][1]
        logit = outputs[output][2][:, 1:]
        frame_label = dataset.get_gt_frame_label(idx)
        new_label = anchor_expand( #这句应该是论文中把标注帧向两边扩展
            logit, cur_label, centers, pv=pv, radious=radious)
        dataset.update_frame_label(idx, new_label)
        new_label = dataset.get_frame_label(idx)
        for t, (ps, gs) in enumerate(zip(new_label, frame_label)):
            frame_level_count += 1
            if ps == gs:
                frame_level_right += 1
            for g in gs:
                gt_count[g] += 1
            if len(new_label[t]) == 0:
                continue
            expand_count += 1
            for p in ps:
                count[p] += 1
                if p in gs:
                    right[p] += 1
    logger.info("correct mined frames for each class: " + ', '.join(map(str, right)))
    logger.info("total mined frames fpr each class: " + ', '.join(map(str, count)))
    count[count == 0] += 1e-3
    logger.info("mined precision for each class" + ', '\
        .join(map(lambda x: str('%.2f' % x), right / count)))
    logger.info("overall correct count, total count, and precision: {}, {}, {}" \
        .format(np.sum(right), np.sum(count), round(np.mean(right / count), 3)))
    logger.info("overall gt count, recall: {}, {}"\
        .format(np.sum(gt_count), round(np.mean(right / gt_count), 3)))
    logger.info("overall total count, accuracy: {}, {}"\
        .format(frame_level_count, frame_level_right / frame_level_count))
    
    dataset.update_num_frames()
   