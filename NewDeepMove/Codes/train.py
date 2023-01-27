from __future__ import division
from __future__ import print_function

import pickle as pickle
import time
from collections import deque, Counter
from json import encoder
import numpy as np
import torch
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
from torch.autograd import Variable
from statistics import mean


class RnnParameterData(object):
    def __init__(self, loc_emb_size=500, uid_emb_size=40, voc_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                 history_mode='avg', attn_type='dot', epoch_max=30, rnn_type='LSTM', model_mode="simple",
                 data_path='./data/', save_path='../results/', data_name='foursquare'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        with open(self.data_path + self.data_name + '.pk', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.vid_list = data['vid_list']
            self.uid_list = data['uid_list']
            self.data_neural = data['data_neural']
            self.tim_size = 48
            self.loc_size = len(self.vid_list)
            self.uid_size = len(self.uid_list)
            self.loc_emb_size = loc_emb_size
            self.tim_emb_size = tim_emb_size
            self.voc_emb_size = voc_emb_size
            self.uid_emb_size = uid_emb_size
            self.hidden_size = hidden_size
            self.epoch = epoch_max
            self.dropout_p = dropout_p
            self.use_cuda = True
            self.lr = lr
            self.lr_step = lr_step
            self.lr_decay = lr_decay
            self.optim = optim
            self.L2 = L2
            self.clip = clip
            self.attn_type = attn_type
            self.rnn_type = rnn_type
            self.history_mode = history_mode
            self.model_mode = model_mode


def generate_input_history(data_neural, mode, mode2=None, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)

            # merge traces with same time stamp
            if mode2 == 'max':
                history_tmp = {}
                for tr in history:
                    if tr[1] not in history_tmp:
                        history_tmp[tr[1]] = [tr[0]]
                    else:
                        history_tmp[tr[1]].append(tr[0])
                history_filter = []
                for t in history_tmp:
                    if len(history_tmp[t]) == 1:
                        history_filter.append((history_tmp[t][0], t))
                    else:
                        tmp = Counter(history_tmp[t]).most_common()
                        if tmp[0][1] > 1:
                            history_filter.append((history_tmp[t][0], t))
                        else:
                            ti = np.random.randint(len(tmp))
                            history_filter.append((tmp[ti][0], t))
                history = history_filter
                history = sorted(history, key=lambda x: x[1], reverse=False)
            elif mode2 == 'avg':
                history_tim = [t[1] for t in history]
                history_count = [1]
                last_t = history_tim[0]
                count = 1
                for t in history_tim[1:]:
                    if t == last_t:
                        count += 1
                    else:
                        history_count[-1] = count
                        history_count.append(1)
                        last_t = t
                        count = 1
            ################

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            if mode2 == 'avg':
                trace['history_count'] = history_count

            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_input_long_history2(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        trace = {}
        session = []
        for c, i in enumerate(train_id):
            session.extend(sessions[i])
        target = np.array([s[0] for s in session[1:]])

        loc_tim = []
        loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
        loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
        tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
        trace['loc'] = Variable(torch.LongTensor(loc_np))
        trace['tim'] = Variable(torch.LongTensor(tim_np))
        trace['target'] = Variable(torch.LongTensor(target))
        data_train[u][i] = trace
        if mode == 'train':
            train_idx[u] = [0, i]
        else:
            train_idx[u] = [i]
    return data_train, train_idx


def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])

            history_tim = [t[1] for t in history]
            history_count = [1]
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_count'] = history_count

            loc_tim = history
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target))
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_queue(train_idx, mode, mode2):

    user = list(train_idx.keys())
    print('len of users in queue :{}'.format(len(user)))
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        if mode2 == 'train':
            for u in user:
                initial_queue[u] = deque(train_idx[u][1:])
        else:
            for u in user:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = {'top1': 0, 'top5': 0, 'top10': 0}
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc['top10'] += 1
        if t in p[:5] and t > 0:
            acc['top5'] += 1
        if t == p[0] and t > 0:
            acc['top1'] += 1
    return acc


def run_simple(data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2=None):
    run_queue = None
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)
    print("queue_len: {}".format(queue_len))
    users_acc = {}
    for c in range(queue_len):
        if c % 100 == 0:
            print("c = {}".format(c))
        st = time.time()
        optimizer.zero_grad()

        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0, 0, 0]
        loc = data[u][i]['loc'].cuda()
        tim = data[u][i]['tim'].cuda()
        target = data[u][i]['target'].cuda()
        uid = Variable(torch.LongTensor([u])).cuda()

        if 'attn' in mode2:
            history_loc = data[u][i]['history_loc'].cuda()
            history_tim = data[u][i]['history_tim'].cuda()

        if mode2 in ['simple', 'simple_long']:
            scores = model(loc, tim)
        elif mode2 == 'attn_avg_long_user':
            history_count = data[u][i]['history_count']
            target_len = target.data.size()[0]
            scores = model(loc, tim, history_loc, history_tim, history_count, uid, target_len)
        elif mode2 == 'attn_local_long':
            target_len = target.data.size()[0]
            scores = model(loc, tim, target_len)

        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]
        loss = criterion(scores, target)

        if mode == 'train':
            loss.backward()
            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
        elif mode == 'test':
            users_acc[u][0] += len(target)
            # Scores are calculated for each user
            acc = get_acc(target, scores)

            _, labels__ = torch.max(scores, dim=1)
            # Top 10
            users_acc[u][3] += acc['top10']
            # Top 5
            users_acc[u][2] += acc['top5']

            # Top 1
            users_acc[u][1] += acc['top1']

        total_loss.append(loss.data.cpu().numpy())
    # pdb.set_trace()
    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss
    elif mode == 'test':

        users_rnn_acc = {}
        accs_10_5_1 = {}

        for u in users_acc:
            accs_10_5_1[u] = (np.array(users_acc[u][1:]) / users_acc[u][0]).tolist()
        avg_acc_10 = mean([accs_10_5_1[x][2] for x in accs_10_5_1])
        avg_acc_5 = mean([accs_10_5_1[x][1] for x in accs_10_5_1])
        avg_acc_1 = mean([accs_10_5_1[x][0] for x in accs_10_5_1])
        print('acc:@1 {},@5 {},@10 {}'.format(avg_acc_1, avg_acc_5, avg_acc_10))
        return avg_loss, (avg_acc_1, avg_acc_5, avg_acc_10), users_rnn_acc
