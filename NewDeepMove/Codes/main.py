import json
import os
import random

import easydict as easydict
import numpy as np
import torch
import time

from torch import nn, optim

from NewDeepMove.Codes.model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong
from NewDeepMove.Codes.train import RnnParameterData, generate_input_history, generate_input_long_history2, \
    generate_input_long_history, run_simple


def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path)
    argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}
    print('*' * 15 + 'start training' + '*' * 15)
    print('model_mode:{} history_mode:{} users:{}'.format(
        parameters.model_mode, parameters.history_mode, parameters.uid_size))

    if parameters.model_mode in ['simple', 'simple_long']:
        model = TrajPreSimple(parameters=parameters).cuda()
    elif parameters.model_mode == 'attn_avg_long_user':
        print("long user\n")
        model = TrajPreAttnAvgLongUser(parameters=parameters).cuda()
    elif parameters.model_mode == 'attn_local_long':
        model = TrajPreLocalAttnLong(parameters=parameters).cuda()
    if args.pretrain == 1:
        model.load_state_dict(torch.load("./pretrain/" + args.model_mode + "/res.m"))

    if 'max' in parameters.model_mode:
        parameters.history_mode = 'max'
    elif 'avg' in parameters.model_mode:
        parameters.history_mode = 'avg'
    else:
        parameters.history_mode = 'whole'

    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=parameters.lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
    #                                                  factor=parameters.lr_decay, threshold=1e-3)


    lr = parameters.lr
    metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [],'accuracy5': [],'accuracy10': [], 'valid_acc': {}}
   
    candidate = parameters.data_neural.keys()
    candidate=random.sample(candidate, 700)

    if 'long' in parameters.model_mode:
        long_history = True
    else:
        long_history = False

    if long_history is False:
        data_train, train_idx = generate_input_history(parameters.data_neural, 'train', mode2=parameters.history_mode,
                                                       candidate=candidate)
        
        data_test, test_idx = generate_input_history(parameters.data_neural, 'test', mode2=parameters.history_mode,
                                                     candidate=candidate)
        
    elif long_history is True:
        if parameters.model_mode == 'simple_long':
            data_train, train_idx = generate_input_long_history2(parameters.data_neural, 'train', candidate=candidate)
            data_test, test_idx = generate_input_long_history2(parameters.data_neural, 'test', candidate=candidate)
        else:
            data_train, train_idx = generate_input_long_history(parameters.data_neural, 'train', candidate=candidate)
            data_test, test_idx = generate_input_long_history(parameters.data_neural, 'test', candidate=candidate)

    print('users:{}  train:{} test:{}'.format(len(candidate),
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    SAVE_PATH = args.save_path
    tmp_path = 'checkpoint/'
    
    if not os.path.exists(SAVE_PATH + tmp_path):
        os.makedirs(SAVE_PATH + tmp_path)
    for epoch in range(parameters.epoch):
        st = time.time()
        if args.pretrain == 0:
            model, avg_loss = run_simple(data_train, train_idx, 'train', lr, parameters.clip, model, optimizer,
                                         criterion, parameters.model_mode)
            print('==>Train Epoch:{:0>2d} Loss:{:.4f}'.format(epoch, avg_loss))
            metrics['train_loss'].append(avg_loss)

        # avg_loss, avg_acc, users_acc = run_simple(data_test, test_idx, 'test', lr, parameters.clip, model,
        #                                           optimizer, criterion, parameters.model_mode)
        avg_loss, (avg_acc_1,avg_acc_5,avg_acc_10), users_acc=run_simple(data_test, test_idx, 'test', lr, parameters.clip, model,
                                                  optimizer, criterion, parameters.model_mode)
        print('==>Test Acc1:{},Test Acc5:{},Test Acc10:{} Loss:{:.4f}'.format(avg_acc_1,avg_acc_5,avg_acc_10, avg_loss))

        metrics['valid_loss'].append(avg_loss)
        metrics['accuracy'].append(avg_acc_1)
        metrics['accuracy5'].append(avg_acc_5)
        metrics['accuracy10'].append(avg_acc_10)
        metrics['valid_acc'][epoch] = users_acc

        save_name_tmp = 'ep_' + str(epoch) + '.pt'
        torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)

        
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            load_epoch = np.argmax(metrics['accuracy'])
            load_name_tmp = 'ep_' + str(load_epoch) + '.pt'
            model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
            print('load epoch={} model state'.format(load_epoch))
        if epoch == 0:
            print('single epoch time cost:{}'.format(time.time() - st))
        if lr <= 0.9 * 1e-5:
            break
        if args.pretrain == 1:
            break

    mid = np.argmax(metrics['accuracy'])
    avg_acc = metrics['accuracy'][mid]
    avg_acc_5=metrics['accuracy5'][mid]
    avg_acc_10=metrics['accuracy10'][mid]
    load_name_tmp = 'ep_' + str(mid) + '.pt'
    model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
    save_name = 'res'
    json.dump({'args': argv, 'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)
    metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'accuracy5': [], 'accuracy10': []}
    for key in metrics_view:
        metrics_view[key] = metrics[key]
    print('*'*10)    
    print(metrics_view)
    print('*'*10)    
    json.dump({'args': argv, 'metrics': metrics_view}, fp=open(SAVE_PATH + save_name + '.txt', 'w'), indent=4)
    torch.save(model.state_dict(), SAVE_PATH + save_name + '.pt')

    for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
        for name in files:
            remove_path = os.path.join(rt, name)
            os.remove(remove_path)
    os.rmdir(SAVE_PATH + tmp_path)

    return avg_acc


def load_pretrained_model(config):
    res = json.load(open("./pretrain/" + config.model_mode + "/res.txt"))
    args = Settings(config, res["args"])
    return args


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["data_name"]
        self.epoch_max = res["epoch_max"]
        self.learning_rate = res["learning_rate"]
        self.lr_step = res["lr_step"]
        self.lr_decay = res["lr_decay"]
        self.clip = res["clip"]
        self.dropout_p = res["dropout_p"]
        self.rnn_type = res["rnn_type"]
        self.attn_type = res["attn_type"]
        self.L2 = res["L2"]
        self.history_mode = res["history_mode"]
        self.model_mode = res["model_mode"]
        self.optim = res["optim"]
        self.hidden_size = res["hidden_size"]
        self.tim_emb_size = res["tim_emb_size"]
        self.loc_emb_size = res["loc_emb_size"]
        self.uid_emb_size = res["uid_emb_size"]
        self.voc_emb_size = res["voc_emb_size"]
        self.pretrain = 1




if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"    
    args= easydict.EasyDict({
    "loc_emb_size"  :50,
    "uid_emb_size":  20,
    "voc_emb_size":50,
    "tim_emb_size":10,
    "hidden_size":80,
    "dropout_p" :0.5,
    "data_name":"foursquare_nyc",
    "learning_rate":0.002,
    "lr_step":1.0,
    "lr_decay":0.1,
    "optim":"Adam",
    "L2" :1 * 1e-6, 
    "clip" :1.0,
    "epoch_max":10,
    "history_mode":"avg",
    "rnn_type":"GRU", 
    "attn_type":"dot",
    "data_path":"../input/foursquare-nyc/",
    "save_path":"../output/kaggle/working/",
    "model_mode":"attn_local_long",
    "accuracy_mode":"top1",
    "pretrain":0})
    print(args)
    if args.pretrain == 1:
        args = load_pretrained_model(args)

    ours_acc = run(args)
