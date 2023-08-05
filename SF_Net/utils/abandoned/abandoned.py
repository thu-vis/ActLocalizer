
torch.set_default_tensor_type('torch.cuda.FloatTensor')
args = options.parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# specifying parameters for debug
if args.debug:
    args.choice = 3
    args.active_ratio = 0.1
    args.resume = "/data/changjian/VideoModel/SF-Net/result/Thumos19/data_Thumos19_mode_active-0.1_lr_0.0001_br_7_alpha_1_beta_1_choice_3_prop_1/ckpt/sfnet.2000.pkl"
    # args.resume = "/data/changjian/VideoVis/SF-Net/result/Thumos19/data_Thumos19_mode_single_lr_0.0001_br_7_alpha_1_beta_1_choice_3_prop_0/ckpt/sfnet.2000.pkl"
    args.mode = "active"

def main():
    basic_config = config.data_config[args.dataset_name]
    # print(basic_config)
    threshold_type = basic_config['threshold_type']
    prediction_filename = basic_config["prediction_filename"]
    feature_dim = basic_config["feature_dim"]
    tiou_thresholds = basic_config["tiou_thresholds"]
    train_subset = basic_config["train_subset"]
    test_subset = basic_config["test_subset"]
    fps = basic_config["fps"]
    stride = basic_config["stride"]
    t_max = basic_config["t_max"]
    t_max_ctc = basic_config["t_max_ctc"]
    num_class = basic_config["num_class"]
    groundtruth_filename = basic_config["groundtruth_filename"]

    device = torch.device("cuda")
    torch.cuda.set_device(2)
    if args.background:
        num_class += 1

    args.start_iter = 0

    dataset = Dataset(args,
                      groundtruth_filename,
                      train_subset=train_subset,
                      test_subset=test_subset,
                      mode=args.mode,
                      use_sf=args.use_sf,
                      choice=args.choice)

    # solve args info and generate dirs
    mode = args.mode
    if mode == "active":
        mode = mode + "-" + str(args.active_ratio)
    args_info = 'data_{}_mode_{}_lr_{}_br_{}_alpha_{}_beta_{}_choice_{}'\
            .format(args.dataset_name, args.mode, args.lr, args.tm, args.alpha, args.beta, str(args.choice))
    # args_info = 'choice_' + str(args.choice)
    args.model_dir = './result/' + args.dataset_name + '/' + args_info + '/ckpt'
    args.log_dir = './result/' + args.dataset_name + '/' + args_info + '/logs'
    prediction_filename = './result/' + args.dataset_name + '/' + args_info + '/prediction.json'

    os.system('mkdir -p %s' % args.model_dir)
    # os.system('mkdir -p %s/%s' % (args.log_dir, args.model_name))
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    tb_writer = TB_Writer('%s' % (args.log_dir))

    # init logger file
    logger = get_logger(args.log_dir)

    logger.info("test")
    logger.info(' ' + args.dataset_name + ' ' + args_info)

    model = SFNET(dataset.feature_size, num_class).to(device)
    iteration, model_state_dict, optimizer_state_dict,\
        optimizer_centloss_f_state_dict, \
        optimizer_centloss_r_state_dict = \
        resume_checkpoint(args.resume, args.model_dir)
    if iteration > 0:
        args.start_iter = iteration
        model.load_state_dict(model_state_dict)
    else:
        logger.info("random initialization!!")


    if args.eval_only and args.resume is None:
        print('***************************')
        print('Pretrained Model NOT Loaded')
        print('Evaluating on Random Model')
        print('***************************')

    act_expand(args,
        dataset,
        model,
        device,
        centers=None)

def detailed_min_distance(ac_features, rt_features):
    simi = np.dot(ac_features, rt_features.T)
    g = nx.DiGraph()
    g.add_node("source", demand=-1)
    g.add_node("sink", demand=1)
    k = 3
    neighs = []
    order = {i : None for i in range(len(rt_features))}
    for i in range(len(ac_features)):
        d = simi[i, :]
        sorted_idxs = d.argsort()[::-1]
        neighs.append(sorted_idxs[:k].tolist())

        for idx, j in enumerate(neighs[-1]):
            name = str(i) + "-" + str(j)
            for t in range(j + 1):
                node = order[t]
                if node and node[0] != name:
                    g.add_edge(node[0], name, weight= - node[1], capacity=1)
        
        for idx, j in enumerate(neighs[-1]):
            name = str(i) + "-" + str(j)
            order[j] = [name, d[j]]
            g.add_edge("source", name, weight=0, capacity=1)
            g.add_edge(name, "sink", weight= -d[j], capacity=1)
        
            # # multiple to multiple
            # for _idx, _j in enumerate(neighs[-1]):
            #     _name = str(i) + "-" + str(_j)
            #     if idx >= _idx:
            #         continue
            #     if j > _j:
            #         g.add_edge(_name, name, weight= - d[_j], capacity=1)
    # pos = nx.circular_layout(g)
    # nx.draw(g, pos, with_labels=True)
    # plt.savefig("./test/active_test/graph.jpg")
    flowDict = nx.min_cost_flow(g)
    

    # process flowDict
    pairs = []
    node = "source"
    while node != "sink":
        node = [n for n in flowDict[node] if flowDict[node][n] == 1][0]
        pairs.append(node)
    cost = 0
    for pair in pairs[:-1]:
        s = int(pair.split("-")[0])
        e = int(pair.split("-")[1])
        cost += simi[s, e]
    
    return pairs[:-1], cost

def match(dataset, ac_features, ac_start, all_features, idvdual_to_whole, idx, frame_id, lb, rb):
    s = lb
    e = rb
    # print("lb", lb, "rb", rb)
    rt_features = []
    for f in range(s, e):
        frame_id = str(idx) + "-" + str(f)
        rt_features.append(all_features[idvdual_to_whole[frame_id]])
    rt_features = np.array(rt_features)
    ac_features = np.array(ac_features)
    pairs, cost = detailed_min_distance(ac_features, rt_features)
    pairs = [(int(i.split("-")[0]) + ac_start, int(i.split("-")[1]) + s) for i in pairs]
    start = min([i[1] for i in pairs])
    end = max([i[1] for i in pairs])
    return start, end, cost

def pair_detailed_min_distance_with_flow(feature_pair_1, feature_pair_2):
    # t0 = time()
    simi = np.dot(feature_pair_1, feature_pair_2.T)
    topk = 3
    nodes = []
    for i in range(len(feature_pair_1)):
        d = simi[i, :]
        sorted_idxs = d.argsort()[::-1][:topk]
        for j in sorted_idxs:
            nodes.append(encoding_pair(i,j))
    for j in range(len(feature_pair_2)):
        d = simi[:, j]
        sorted_idxs = d.argsort()[::-1][:topk]
        for i in sorted_idxs:
            nodes.append(encoding_pair(i,j))
    nodes = list(set(nodes))
    g = nx.DiGraph()
    g.add_node("source", demand=-1)
    g.add_node("sink", demand=1)

    for a in nodes:
        a_1, a_2 = decoding_pair(a)
        w = simi[a_1, a_2]
        for b in nodes:
            if a == b:
                continue
            b_1, b_2 = decoding_pair(b)
            if a_1 <= b_1 and a_2 <= b_2:
                g.add_edge(a, b, weight= - w, capacity=1)
        g.add_edge("source", a, weight=0, capacity=1)
        g.add_edge(a, "sink", weight= -w, capacity=1)
    
    # print("first time: ", time() - t0)

    # solving max flow problem
    flowDict = nx.min_cost_flow(g)
    # process flowDict
    pairs = []
    node = "source"
    while node != "sink":
        node = [n for n in flowDict[node] if flowDict[node][n] == 1][0]
        pairs.append(node)
    # print("second time: ", time() - t0)

    cost = 0
    for pair in pairs[:-1]:
        s = int(pair.split("-")[0])
        e = int(pair.split("-")[1])
        cost += simi[s, e]

    return pairs[:-1], cost

def propagation(dataset, outputs):
    all_features, all_preds, whole_to_invdual, idvdual_to_whole, labeled_frames = get_features(dataset, outputs)
    all_features = feature_norm(all_features)
    print("total features:", len(all_features)) 
    labeled_features = [i["idx"] for i in labeled_frames]
    labeled_features = all_features[np.array(labeled_features)]
    for idx in dataset.trainidx:
    # for idx in [484]:
        active_label = dataset.active_labels[idx]
        print("active label idx", idx)
        for ac in active_label:
            # if ac[2] != 12:
            #     continue
            s = ac[0]
            # e = min(ac[1] + 1, len(dataset.frame_labels[idx]))
            e = min(ac[1], len(dataset.frame_labels[idx]))
            ac_features = []
            for f in range(s, e):
                frame_id = str(idx) + "-" + str(f)
                ac_features.append(all_features[idvdual_to_whole[frame_id]])
            feature = np.array(ac_features).mean(axis=0)
            simi = np.dot(feature[np.newaxis, :], labeled_features.T)
            sorted_idx = simi.reshape(-1).argsort()[::-1]
            count = 0
            for ci, i in enumerate(sorted_idx):
                labeled_label = labeled_frames[i]["label"]
                if ac[2] not in labeled_label:
                    continue
                labeled_id = labeled_frames[i]["id"]
                labeled_idx = labeled_frames[i]["idx"]
                labeled_video_id, labeled_frame_id = labeled_id.split("-")
                labeled_video_id = int(labeled_video_id)
                labeled_frame_id = int(labeled_frame_id)
                # print("labeled_video_id", labeled_video_id, "labeled_video_id", labeled_frame_id)
                if dataset.is_actively_labeled(labeled_video_id, labeled_frame_id):
                    continue
                # left bound to the nearest labeled frame on the left
                lb = left_bound(dataset.frame_labels[labeled_video_id], labeled_frame_id, ac[2])
                # left bound to the nearest labeled frame on the right
                rb = right_bound(dataset.frame_labels[labeled_video_id], labeled_frame_id, ac[2])
                if labeled_video_id != 486 or labeled_frame_id != 9:
                    continue
                if lb == rb:
                    continue
                # link construction
                try:
                    start, end, cost = match(dataset, ac_features, s, all_features, idvdual_to_whole, \
                        labeled_video_id, labeled_frame_id, lb, rb)
                except:
                    import IPython; IPython.embed(); exit()
                # print("{}\t{}\t{}\t{}\t{}\t{}\t{}"\
                #     .format(ci, labeled_video_id, lb, labeled_frame_id, rb, start, end))
                print("{}\t{}\t{}\t{}".format(labeled_video_id, labeled_frame_id, simi[0][i], cost))
                # propagation 
                outputs[labeled_video_id][2][start: end, ac[2]] = \
                    outputs[labeled_video_id][2][labeled_frame_id, ac[2]]
                count += 1
                if count >= 5:
                    break
    exit()

def act_expand(args, dataset, model, device, radious=3, pv=0.95, centers=None, prop=False):
    logger = get_logger()
    classlist = dataset.get_classlist()
    right = np.zeros(len(classlist))
    count = np.zeros(len(classlist))
    gt_count = np.zeros(len(classlist))
    # Batch fprop
    train_idx = dataset.get_trainidx()
    expand_count = 0
    classlist = dataset.get_classlist()
    centers = [[] for _ in range(len(classlist))]
    # outputs = []
    outputs = {}
    for idx in train_idx:
        feat = dataset.get_feature(idx)
        feat = torch.from_numpy(np.expand_dims(feat,
                                               axis=0)).float().to(device)
        cur_label = dataset.get_init_frame_label(idx)
        with torch.no_grad():
            _, logits_f, _, logits_r, tcam, _, _, _ = model(
                Variable(feat), device, is_training=False)
            tcam = tcam.data.cpu().numpy().squeeze()
            # from IPython import embed
            # embed()
            # if args.background:
            #     tcam = tcam[:, 1:]
            assert len(cur_label) == len(tcam)
            for jdx, ls in enumerate(cur_label):
                if len(ls) > 0:
                    for l in ls:
                        centers[l].append(tcam[jdx])
                if dataset.is_actively_labeled(idx, jdx):
                    cur_label[jdx] = []
            # outputs += [[idx, cur_label, tcam]]
            outputs[idx] = [idx, cur_label, tcam]
    # propagation_base_on_graph(dataset, outputs)
    for output in outputs:
        idx = output
        cur_label = outputs[output][1]
        logit = outputs[output][2]
        frame_label = dataset.get_gt_frame_label(idx)
        new_label = anchor_expand(
            logit, cur_label, centers, pv=pv, radious=radious)
        dataset.update_frame_label(idx, new_label)
        new_label = dataset.get_frame_label(idx)
        for t, (ps, gs) in enumerate(zip(new_label, frame_label)):
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
    dataset.update_num_frames()

if __name__ == '__main__':
    main()

def propagation(dataset, outputs):
    all_features, whole_to_invdual, idvdual_to_whole, labeled_frames = get_features(dataset, outputs)
    all_features = feature_norm(all_features)
    print("total features:", len(all_features))
    labeled_features = [i["idx"] for i in labeled_frames]
    labeled_features = all_features[np.array(labeled_features)]
    for idx in dataset.trainidx:
    # for idx in [484]:
        active_label = dataset.active_labels[idx]
        print("active label idx", idx)
        for ac in active_label:
            # if ac[2] != 12:
            #     continue
            s = ac[0]
            # e = min(ac[1] + 1, len(dataset.frame_labels[idx]))
            e = min(ac[1], len(dataset.frame_labels[idx]))
            ac_features = []
            for f in range(s, e):
                frame_id = str(idx) + "-" + str(f)
                ac_features.append(all_features[idvdual_to_whole[frame_id]])
            feature = np.array(ac_features).mean(axis=0)
            simi = np.dot(feature[np.newaxis, :], labeled_features.T)
            sorted_idx = simi.reshape(-1).argsort()[::-1]
            count = 0
            for ci, i in enumerate(sorted_idx):
                labeled_label = labeled_frames[i]["label"]
                if ac[2] not in labeled_label:
                    continue
                labeled_id = labeled_frames[i]["id"]
                labeled_idx = labeled_frames[i]["idx"]
                labeled_video_id, labeled_frame_id = labeled_id.split("-")
                labeled_video_id = int(labeled_video_id)
                labeled_frame_id = int(labeled_frame_id)
                # print("labeled_video_id", labeled_video_id, "labeled_video_id", labeled_frame_id)
                if dataset.is_actively_labeled(labeled_video_id, labeled_frame_id):
                    continue
                # left bound to the nearest labeled frame on the left
                lb = left_bound(dataset.frame_labels[labeled_video_id], labeled_frame_id, ac[2])
                # left bound to the nearest labeled frame on the right
                rb = right_bound(dataset.frame_labels[labeled_video_id], labeled_frame_id, ac[2])
                # if labeled_video_id != 489 or labeled_frame_id != 488:
                #     continue
                if lb == rb:
                    continue
                # link construction
                start, end, cost = match(dataset, ac_features, s, all_features, idvdual_to_whole, \
                    labeled_video_id, labeled_frame_id, lb, rb)

                print("{}\t{}\t{}\t{}".format(labeled_video_id, labeled_frame_id, simi[0][i], cost))
                # propagation 
                outputs[labeled_video_id][2][start: end, ac[2]] = \
                    outputs[labeled_video_id][2][labeled_frame_id, ac[2]]
                count += 1
                if count >= 3:
                    break

def act_expand_v1(args, dataset, model, device, radious=3, pv=0.95, centers=None, prop=False):
    logger = get_logger()
    classlist = dataset.get_classlist()
    right = np.zeros(len(classlist))
    count = np.zeros(len(classlist))
    gt_count = np.zeros(len(classlist))
    # Batch fprop
    train_idx = dataset.get_trainidx()
    expand_count = 0
    classlist = dataset.get_classlist()
    centers = [[] for _ in range(len(classlist))]
    outputs = []
    for idx in train_idx:
        feat = dataset.get_feature(idx)
        feat = torch.from_numpy(np.expand_dims(feat,
                                               axis=0)).float().to(device)
        cur_label = dataset.get_init_frame_label(idx)
        with torch.no_grad():
            _, logits_f, _, logits_r, tcam, _, _, _ = model(
                Variable(feat), device, is_training=False)
            tcam = tcam.data.cpu().numpy().squeeze()
            if args.background:
                tcam = tcam[:, 1:]
            assert len(cur_label) == len(tcam)
            for jdx, ls in enumerate(cur_label):
                if len(ls) > 0:
                    for l in ls:
                        centers[l].append(tcam[jdx])
                if dataset.is_actively_labeled(idx, jdx):
                    cur_label[jdx] = []
            outputs += [[idx, cur_label, tcam]]
    
    for output in outputs:
        idx = output[0]
        cur_label = output[1]
        logit = output[2]
        frame_label = dataset.get_gt_frame_label(idx)
        new_label = anchor_expand(
            logit, cur_label, centers, pv=pv, radious=radious)
        dataset.update_frame_label(idx, new_label)
        new_label = dataset.get_frame_label(idx)
        for t, (ps, gs) in enumerate(zip(new_label, frame_label)):
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
    dataset.update_num_frames()

def act_expand_finetune(args, dataset, model, device, radious=3, pv=0.95, centers=None, prop=False,
    alpha_1=0.5, alpha_2=1, beta=0.5):

    classlist = dataset.get_classlist()
    right = np.zeros(len(classlist))
    count = np.zeros(len(classlist))
    gt_count = np.zeros(len(classlist))
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
        cur_label = dataset.get_init_frame_label(idx) # cur_label应该指的是有标注的帧吧
        with torch.no_grad():
            _, logits_f, _, logits_r, tcam, _, _, _ = model(
                Variable(feat), device, is_training=False)
            tcam = tcam.data.cpu().numpy().squeeze()
            if args.background: # 为True的话，把背景帧去掉了？？
                tcam = tcam[:, 1:]
            assert len(cur_label) == len(tcam)
            for jdx, ls in enumerate(cur_label):
                if len(ls) > 0:
                    for l in ls:
                        centers[l].append(tcam[jdx]) # centers[l]中存的是标注为l的所有样本
                if dataset.is_actively_labeled(idx, jdx): #这句是啥意思
                    cur_label[jdx] = []
            # outputs += [[idx, cur_label, tcam]]
            outputs[idx] = [idx, cur_label, tcam] # idx: 
    if prop:
        propagation_base_on_graph(dataset, outputs, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta)
    for output in outputs:
        idx = output
        cur_label = outputs[output][1]
        logit = outputs[output][2]
        frame_label = dataset.get_gt_frame_label(idx)
        new_label = anchor_expand( #这句应该是论文中把标注帧向两边扩展
            logit, cur_label, centers, pv=pv, radious=radious)
        dataset.update_frame_label(idx, new_label)
        new_label = dataset.get_frame_label(idx)
        for t, (ps, gs) in enumerate(zip(new_label, frame_label)):
            for g in gs:
                gt_count[g] += 1
            if len(new_label[t]) == 0:
                continue
            expand_count += 1
            for p in ps:
                count[p] += 1
                if p in gs:
                    right[p] += 1
    return np.sum(right), np.sum(count), round(np.sum(right) / np.sum(count), 3), round(np.mean(right / count), 3)
 
def pair_detailed_min_distance_with_flow(feature_pair_1, feature_pair_2):
    # t0 = time()
    simi = np.dot(feature_pair_1, feature_pair_2.T)
    topk = 3
    nodes = []
    for i in range(len(feature_pair_1)):
        d = simi[i, :]
        sorted_idxs = d.argsort()[::-1][:topk]
        for j in sorted_idxs:
            nodes.append(encoding_pair(i,j))
    for j in range(len(feature_pair_2)):
        d = simi[:, j]
        sorted_idxs = d.argsort()[::-1][:topk]
        for i in sorted_idxs:
            nodes.append(encoding_pair(i,j))
    nodes = list(set(nodes))
    g = nx.DiGraph()
    g.add_node("source", demand=-1)
    g.add_node("sink", demand=1)

    for a in nodes:
        a_1, a_2 = decoding_pair(a)
        w = simi[a_1, a_2]
        for b in nodes:
            if a == b:
                continue
            b_1, b_2 = decoding_pair(b)
            if a_1 <= b_1 and a_2 <= b_2:
                g.add_edge(a, b, weight= - w, capacity=1)
        g.add_edge("source", a, weight=0, capacity=1)
        g.add_edge(a, "sink", weight= -w, capacity=1)
    
    # print("first time: ", time() - t0)

    # solving max flow problem
    flowDict = nx.min_cost_flow(g)
    # process flowDict
    pairs = []
    node = "source"
    while node != "sink":
        node = [n for n in flowDict[node] if flowDict[node][n] == 1][0]
        pairs.append(node)
    # print("second time: ", time() - t0)

    cost = 0
    for pair in pairs[:-1]:
        s = int(pair.split("-")[0])
        e = int(pair.split("-")[1])
        cost += simi[s, e]

    return pairs[:-1], cost

def pair_detailed_min_distance(feature_pair_1, feature_pair_2):
    # t0 = time()
    simi = np.dot(feature_pair_1, feature_pair_2.T)
    topk = 3
    nodes = []
    for i in range(len(feature_pair_1)):
        d = simi[i, :]
        sorted_idxs = d.argsort()[::-1][:topk]
        for j in sorted_idxs:
            nodes.append(encoding_pair(i,j))
    for j in range(len(feature_pair_2)):
        d = simi[:, j]
        sorted_idxs = d.argsort()[::-1][:topk]
        for i in sorted_idxs:
            nodes.append(encoding_pair(i,j))
    nodes = list(set(nodes))

    t = nx.DiGraph()
    t.add_node("source", demand=-1)
    t.add_node("sink", demand=1)

    for a in nodes:
        a_1, a_2 = decoding_pair(a)
        w = simi[a_1, a_2]
        for b in nodes:
            if a == b:
                continue
            b_1, b_2 = decoding_pair(b)
            if a_1 <= b_1 and a_2 <= b_2:
                t.add_edge(a, b, weight= w, capacity=1)
        t.add_edge("source", a, weight=0, capacity=1)
        t.add_edge(a, "sink", weight= w, capacity=1)
    

    res = nx.dag_longest_path(t)
    pairs = res[1:]

    cost = 0
    for pair in res[1:-1]:
        s = int(pair.split("-")[0])
        e = int(pair.split("-")[1])
        cost += simi[s, e]

    return pairs[:-1], cost

def pair_matching(all_features, idvdual_to_whole, pair_1, pair_2):
    feature_pair_1 = []
    for f in range(pair_1["lb"], pair_1["rb"] + 1):
        frame_id = str(pair_1["v_id"]) + "-" + str(f)
        feature_pair_1.append(all_features[idvdual_to_whole[frame_id]])
    feature_pair_1 = np.array(feature_pair_1)
    feature_pair_2 = []
    for f in range(pair_2["lb"], pair_2["rb"] + 1):
        frame_id = str(pair_2["v_id"]) + "-" + str(f)
        feature_pair_2.append(all_features[idvdual_to_whole[frame_id]])
    feature_pair_2 = np.array(feature_pair_2)
    pairs, cost = pair_detailed_min_distance(feature_pair_1, feature_pair_2)
    pairs = [(int(i.split("-")[0]) + pair_1["lb"], int(i.split("-")[1]) + pair_2["lb"]) for i in pairs]
    pairs = [(str(pair_1["v_id"]) + "-" + str(p[0]), str(pair_2["v_id"]) + "-" + str(p[1])) for p in pairs]
    return pairs, cost

def graph_construction(dataset, all_features, idvdual_to_whole, labeled_frames):
    # all_features, whole_to_invdual, idvdual_to_whole, labeled_frames = get_features(dataset)
    # all_features = feature_norm(all_features)
    # print("total features:", len(all_features))
    labeled_features = [i["idx"] for i in labeled_frames]
    labeled_features = all_features[np.array(labeled_features)]
    # aggregate the features of labeled actions
    for idx, labeled_frame in enumerate(labeled_frames):
        labeled_id = labeled_frame["id"]
        labeled_video_id, labeled_frame_id = labeled_id.split("-")
        labeled_video_id = int(labeled_video_id)
        labeled_frame_id = int(labeled_frame_id)
        active_label = dataset.is_actively_labeled(labeled_video_id, labeled_frame_id)
        if active_label:
            s = active_label[0]
            e = min(active_label[1] + 1, len(dataset.frame_labels[labeled_video_id]))
            ac_features = []
            for f in range(s, e):
                frame_id = str(labeled_video_id) + "-" + str(f)
                ac_features.append(all_features[idvdual_to_whole[frame_id]])
            feature = np.array(ac_features).mean(axis=0)
            labeled_features[idx] = feature
        
    # construction
    topk = 3
    labeled_distance = pairwise_distances(labeled_features)
    edges = []
    for count_idx, labeled_frame in tqdm(enumerate(labeled_frames)):
    # for count_idx in [19]:
    #     labeled_frame = labeled_frames[count_idx]
        print("count_idx: ", count_idx)
        labeled_id = labeled_frame["id"]
        labeled_idx = labeled_frame["idx"]
        labeled_label = labeled_frame["label"][0] # TODO: careful
        labeled_video_id, labeled_frame_id = labeled_id.split("-")
        labeled_video_id = int(labeled_video_id)
        labeled_frame_id = int(labeled_frame_id)
        labeled_lb, labeled_rb, labeled_is_active = acquire_boundaries(dataset, \
            labeled_video_id, labeled_frame_id, labeled_label)
        labeled_pair = {
            "lb": labeled_lb,
            "rb": labeled_rb,
            "v_id": labeled_video_id
        }
        dis = labeled_distance[count_idx]
        labels = []
        sorted_idx = dis.argsort()
        for idx in sorted_idx:
            if count_idx == idx:
                continue
            print("idx: ", idx)
            fetched_id = labeled_frames[idx]["id"]
            fetched_label = labeled_frames[idx]["label"][0]
            if fetched_label != labeled_label:
                continue
            fetched_video_id, fetched_frame_id = fetched_id.split("-")
            fetched_video_id = int(fetched_video_id)
            fetched_frame_id = int(fetched_frame_id)
            fetched_lb, fetched_rb, fetched_labeled_is_active = acquire_boundaries(dataset, \
            fetched_video_id, fetched_frame_id, labeled_label)
            fetched_pair = {
                "lb": fetched_lb,
                "rb": fetched_rb,
                "v_id": fetched_video_id
            }
            pairs, cost = pair_matching(all_features, idvdual_to_whole, labeled_pair, fetched_pair)
            pairs = [(idvdual_to_whole[p[0]], idvdual_to_whole[p[1]]) for p in pairs]
            edges += pairs
            labels.append(idx)
            if len(labels) >= topk:
                break
    print("******************* final *******************")
    return construct_csr_matrix_by_edge(edges, all_features.shape[0])

def propagation_2(dataset, outputs):
    all_features, all_preds, whole_to_invdual, idvdual_to_whole, \
        labeled_frames = get_features(dataset, outputs)
    all_features = feature_norm(all_features)
    print("total features:", len(all_features))
    # construction
    affinity_matrix = graph_construction(dataset, all_features, idvdual_to_whole, labeled_frames)
    class_num = len(dataset.classlist)
    fake_center_map = np.random.rand(all_features.shape[0])
    fake_center_map = fake_center_map * (all_features.shape[0] - 1)
    fake_center_map = fake_center_map.astype(int)
    train_y = np.ones((all_features.shape[0], class_num)) * -1
    graph_propagation(affinity_matrix, train_y, fake_center_map)

def min_distance(ac_features, rt_features):
    import networkx as nx
    simi = np.dot(ac_features, rt_features.T)
    g = nx.DiGraph()
    g.add_node("source", demand=-1)
    g.add_node("sink", demand=1)
    k = 3
    neighs = []
    order = {i : None for i in range(len(rt_features))}
    for i in range(len(ac_features)):
        d = simi[i, :]
        sorted_idxs = d.argsort()[::-1]
        neighs.append(sorted_idxs[:k].tolist())

        for idx, j in enumerate(neighs[-1]):
            name = str(i) + "-" + str(j)
            for t in range(j + 1):
                node = order[t]
                if node and node[0] != name:
                    g.add_edge(node[0], name, weight= - node[1], capacity=1)
        
        for idx, j in enumerate(neighs[-1]):
            name = str(i) + "-" + str(j)
            order[j] = [name, d[j]]
            g.add_edge("source", name, weight=0, capacity=1)
            g.add_edge(name, "sink", weight= -d[j], capacity=1)
        
            # # multiple to multiple
            # for _idx, _j in enumerate(neighs[-1]):
            #     _name = str(i) + "-" + str(_j)
            #     if idx >= _idx:
            #         continue
            #     if j > _j:
            #         g.add_edge(_name, name, weight= - d[_j], capacity=1)

    # pos = nx.circular_layout(g)
    # nx.draw(g, pos, with_labels=True)
    # plt.savefig("./test/active_test/graph.jpg")
    
    flowDict = nx.min_cost_flow(g)

    # process flowDict
    pairs = []
    node = "source"
    while node != "sink":
        node = [n for n in flowDict[node] if flowDict[node][n] == 1][0]
        pairs.append(node)
    cost = 0
    for pair in pairs[:-1]:
        s = int(pair.split("-")[0])
        e = int(pair.split("-")[1])
        cost += simi[s, e]
    return pairs[:-1], cost

def fully_min_distance(ac_features, rt_features):
    import networkx as nx
    simi = np.dot(ac_features, rt_features.T)
    g = nx.DiGraph()
    g.add_node("source", demand=-1)
    g.add_node("sink", demand=1)
    k = 3
    neighs = []
    order = {i : [] for i in range(len(rt_features))}
    for i in range(len(ac_features)):
        d = simi[i, :]
        sorted_idxs = d.argsort()[::-1]
        neighs.append(sorted_idxs[:k].tolist())

        for idx, j in enumerate(neighs[-1]):
            name = str(i) + "-" + str(j)
            for t in range(j + 1):
                nodes = order[t]
                for node in nodes:
                    if node and node[0] != name:
                        g.add_edge(node[0], name, weight= - node[1], capacity=1)
        
        for idx, j in enumerate(neighs[-1]):
            name = str(i) + "-" + str(j)
            order[j].append([name, d[j]])
            g.add_edge("source", name, weight=0, capacity=1)
            g.add_edge(name, "sink", weight= -d[j], capacity=1)
        
            # # multiple to multiple
            # for _idx, _j in enumerate(neighs[-1]):
            #     _name = str(i) + "-" + str(_j)
            #     if idx >= _idx:
            #         continue
            #     if j > _j:
            #         g.add_edge(_name, name, weight= - d[_j], capacity=1)

    # pos = nx.circular_layout(g)
    # nx.draw(g, pos, with_labels=True)
    # plt.savefig("./test/active_test/graph.jpg")
    flowDict = nx.min_cost_flow(g)

    # process flowDict
    pairs = []
    node = "source"
    while node != "sink":
        node = [n for n in flowDict[node] if flowDict[node][n] == 1][0]
        pairs.append(node)
    cost = 0
    for pair in pairs[:-1]:
        s = int(pair.split("-")[0])
        e = int(pair.split("-")[1])
        cost += simi[s, e]
    return pairs[:-1], cost

def match(dataset, ac_features, ac_start, all_features, idvdual_to_whole, idx, frame_id, lb, rb):
    s = lb
    e = rb
    # print("lb", lb, "rb", rb)
    rt_features = []
    for f in range(s, e):
        frame_id = str(idx) + "-" + str(f)
        rt_features.append(all_features[idvdual_to_whole[frame_id]])
    rt_features = np.array(rt_features)
    ac_features = np.array(ac_features)
    pairs, cost = min_distance(ac_features, rt_features)
    pairs = [(int(i.split("-")[0]) + ac_start, int(i.split("-")[1]) + s) for i in pairs]
    start = min([i[1] for i in pairs])
    end = max([i[1] for i in pairs])
    return start, end, cost

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
            for step in range(radious):
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
        _expand(-1)
        _expand(1)
    return frame_label

def expand_finetune():
    import pickle
    path = "./buffer/act_expand_params.pkl"
    alpha_1s = [0.2, 0.4, 0.6, 0.8, 1, 2, 5, 10]
    alpha_2s = [1, 2, 4]
    betas = [0.2, 0.5, 0.8, 2, 5, 10, 20]
    best = 0
    bestargs = (0,0,0)
    for alpha_1 in alpha_1s:
        for alpha_2 in alpha_2s:
            alpha_2 *= alpha_1
            for beta in betas:
                with open(path, "rb") as f:
                    args, dataset, model, device, radious, pv, centers, prop = pickle.load(f)
                    correct, all, acc, precision = act_expand_finetune(args, dataset, model, device, radious, pv, centers, prop, alpha_1, alpha_2, beta)
                    print(alpha_1, alpha_2, beta, correct, all, acc, precision)
                    if best<precision:
                        best = precision
                        bestargs = (alpha_1, alpha_2, beta, correct, all, acc, precision)
                        print("best: ", alpha_1, alpha_2, beta, correct, all, acc, precision)
    
    
    print("best", bestargs)
    
