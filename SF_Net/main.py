from __future__ import print_function
import torch.optim as optim
import argparse
import os
import torch
from SF_Net.utils.model import SFNET
from SF_Net.utils.dataset import Dataset
from SF_Net.utils.dataset.config import config
from SF_Net.utils.eval import evaluate
from SF_Net.utils.train import *
from tensorboard_logger import Logger as TB_Writer
from datetime import datetime
import SF_Net.options as options
from SF_Net.utils.model import CenterLoss
from SF_Net.utils.eval.export import export_data
from SF_Net.utils import get_logger, save_checkpoint, resume_checkpoint
import json
import pickle

try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
except:
    None
    
args = options.parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(parameters=None):
    basic_config = config.data_config[args.dataset_name]
    # print(basic_config)
    threshold_type = basic_config['threshold_type']
    prediction_filename = basic_config["prediction_filename"]
    # feature_dim = basic_config["feature_dim"]
    feature_dim = int(args.feature_size / 2)
    tiou_thresholds = basic_config["tiou_thresholds"]
    train_subset = basic_config["train_subset"]
    test_subset = basic_config["test_subset"]
    fps = args.fps = basic_config["fps"]
    stride = args.stride = basic_config["stride"]
    t_max = args.t_max = basic_config["t_max"]
    t_max_ctc = args.t_max_cc = basic_config["t_max_ctc"]
    num_class = basic_config["num_class"]
    groundtruth_filename = basic_config["groundtruth_filename"]

    device = torch.device("cuda")
    if parameters is not None:
        if 'cuda' in parameters.keys():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(parameters['cuda'])
        for key in parameters.keys():
            if key in args:
                args.__setattr__(key, parameters[key])
    
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

    if parameters is not None:
        if 'pseudo_label_dir' in parameters.keys():
            # update dataset by pseudo_label
            dir = parameters['pseudo_label_dir']
            filename = os.path.join(dir, 'pseudo_label.pkl')
            train_idx = dataset.get_trainidx()
            pseudo_labels = pickle.load(open(filename, 'rb'))
            for idx in train_idx:
                dataset.update_frame_label(idx, pseudo_labels[idx])        

    # solve args info and generate dirs
    if args.mode == "fully":
        args.expand = False
    mode = args.mode
    if mode == "active":
        mode = mode + "-" + str(args.active_ratio)
    prop = "0"
    if args.prop:
        prop = "1"
    increase = '0'
    if args.increase:
        increase = '1'
    if args.feature_size != 2048:
        prop = prop +  "_fs_" + str(args.feature_size)
    args_info = 'data_{}_mode_{}_lr_{}_br_{}_alpha_{}_beta_{}_choice_{}_prop_{}_seed_{}_increase_{}'\
        .format(args.dataset_name, mode, args.lr, args.tm, args.alpha, args.beta, str(args.choice),
        prop, args.seed, increase)
    # args_info = 'choice_' + str(args.choice)
    # './result/'
    
    args.dir = '/data/haoze/result/' + args.dataset_name + '/' + str(args.expand_step) + '/' + args_info
    args.model_dir = args.dir + '/ckpt'
    args.log_dir = args.dir + '/logs'
    prediction_output = args.dir

    os.system('mkdir -p %s' % args.model_dir)
    # os.system('mkdir -p %s/%s' % (args.log_dir, args.model_name))
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    tb_writer = TB_Writer('%s' % (args.log_dir))

    # init logger file
    logger = get_logger(args.log_dir)

    logger.info(args.dataset_name + ' ' + args_info)
    model = SFNET(dataset.feature_size, num_class).to(device)
    iteration, model_state_dict, optimizer_state_dict,\
        optimizer_centloss_f_state_dict, \
        optimizer_centloss_r_state_dict = \
        resume_checkpoint(args.resume, args.model_dir, args.eval_only)
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

    # if args.resume is not None:
    #     model.load_state_dict(torch.load(args.resume))

    best_acc = 0
    criterion_cent_f = CenterLoss(num_classes=num_class,
                                  feat_dim=feature_dim,
                                  use_gpu=True)
    criterion_cent_r = CenterLoss(num_classes=num_class,
                                  feat_dim=feature_dim,
                                  use_gpu=True)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_centloss_f = torch.optim.SGD(
        criterion_cent_f.parameters(), lr=0.1)
    optimizer_centloss_r = torch.optim.SGD(
        criterion_cent_r.parameters(), lr=0.1)
    if args.start_iter > 0 and not args.eval_only:
        # load optimizer state
        logger.info("load optimizer state!!")
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer_centloss_f.load_state_dict(optimizer_centloss_f_state_dict)
        optimizer_centloss_r.load_state_dict(optimizer_centloss_r_state_dict)


    criterion_cent_all = [criterion_cent_f, criterion_cent_r]
    optimizer_centloss_all = [optimizer_centloss_f, optimizer_centloss_r]
    center_f = criterion_cent_f.get_centers()
    center_r = criterion_cent_r.get_centers()
    centers = [center_f, center_r]
    params = {'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma}

    ce = torch.nn.CrossEntropyLoss().cuda()
    counts = dataset.get_frame_counts()
    print('total %d annotated frames' % counts)

    if args.export:
        export_data(
            args.start_iter,
            dataset,
            args,
            model,
            device,
            fps=fps,
            stride=stride,
        )
        logger.info("Export Done!")
        exit()

    for itr in range(args.start_iter, args.max_iter + 1):
        dataset.t_max = t_max
        if itr % 2 == 0 and itr > 000:
            dataset.t_max = t_max_ctc
        if itr % args.eval_steps == 0 and (not itr == 0 or args.eval_only) \
            and (args.eval_only or itr != args.start_iter):
            # 在训练集上验证
            # subset=train_subset,
            this_subset = test_subset
            if args.test_on_train:
                this_subset = train_subset
                
            print('model_name: %s' % args.model_name)
            acc = evaluate(itr,
                           dataset,
                           model,
                           tb_writer,
                           groundtruth_filename,
                           prediction_output,
                           background=args.background,
                           fps=fps,
                           stride=stride,
                           subset=this_subset,
                           threshold_type=threshold_type,
                           frame_type=args.frame_type,
                           adjust_mean=args.adjust_mean,
                           act_weight=args.actionness_weight,
                           tiou_thresholds=tiou_thresholds,
                           use_anchor=args.use_anchor,
                           save_output=args.save_output)
            # torch.save(model.state_dict(),
            #            '%s/%s.%d.pkl' % (args.model_dir, args.model_name, itr))
            # torch.save(model.state_dict(),
            #             '%s/%s_best.pkl' % (args.model_dir, args.model_name))
            is_best = False
            if acc >= best_acc and not args.eval_only:
                is_best = True
                best_acc = acc
            if not args.eval_only:
                save_checkpoint(itr, model, optimizer, optimizer_centloss_f, \
                    optimizer_centloss_r, args.model_dir, is_best)
        if args.expand and itr == args.expand_step: # question: 这里仅当itr==expand step时才expand，所以一次训练中仅expand一轮？
            act_expand(args,
                          dataset,
                          model,
                          device,
                          centers=None,
                          prop=args.prop)
            model = SFNET(dataset.feature_size, num_class).to(device)
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=0.0005)
            counts = dataset.get_frame_counts()
            print('total %d frames' % counts)
        if args.eval_only:
            print('Done Eval!')
            break
        if not args.eval_only:
            train_SF(itr, dataset, args, model, optimizer, criterion_cent_all,
                     optimizer_centloss_all, tb_writer, device, ce, params, mode=args.mode)

if __name__ == '__main__':
    main()
