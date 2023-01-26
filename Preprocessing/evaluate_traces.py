from __future__ import print_function
from __future__ import division

import time
import argparse
import numpy as np
import pickle
from collections import Counter
from time import strptime
import pdb

def entropy_spatial(sessions):
    locations = {}
    days = sorted(sessions.keys())
    for d in days:
        session = sessions[d]
        for s in session:
            if s[0] not in locations:
                locations[s[0]] = 1
            else:
                locations[s[0]] += 1
    frequency = np.array([locations[loc] for loc in locations])
    frequency = frequency / np.sum(frequency)
    entropy = - np.sum(frequency * np.log(frequency))
    return entropy

class DataFoursquare(object):
    def __init__(self, trace_min=10, global_visit=10, hour_gap=72, min_gap=10, session_min=2, session_max=10,
                 sessions_min=5, train_split=0.8, embedding_len=50):
        tmp_path = "../data/"
        
        # self.save_name = 'foursquare_nyc_20000'
        # self.DATASET_PATH='/home/local/ASUAD/ychen404/Code/DeepMove_new/dataset_tsmc2014/'
        # self.DATASET_NAME='dataset_TSMC2014_NYC_20000.txt'

        self.save_name = 'tweets-cikm_20000'
        self.DATASET_PATH='serm-data/'
        self.DATASET_NAME='tweets-cikm_20000.txt'

        self.trace_len_min = trace_min
        self.location_global_visit_min = global_visit
        self.hour_gap = hour_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min
        self.sessions_count_min = sessions_min
        self.words_embeddings_len = embedding_len

        self.train_split = train_split

        self.data = {}
        self.venues = {}
        self.words_original = []
        self.words_lens = []
        self.dictionary = dict()
        self.words_dict = None
        self.data_filter = {}
        self.user_filter3 = None
        self.uid_list = {}
        self.vid_list = {'unk': [0, -1]}
        self.vid_list_lookup = {}
        self.vid_lookup = {}
        self.pid_loc_lat = {}
        self.data_neural = {}

    # ############# 1. read trajectory data from twitters
    def load_trajectory_from_tweets(self):
        with open(self.DATASET_PATH + self.DATASET_NAME) as fid:
        # with open(self.TWITTER_PATH, 'r') as fid:
            for i, line in enumerate(fid):
                _, uid, _, _, tim, _, _, tweet, pid = line.strip('\r\n').split('')
                # print("pid:{}".format(pid))
                
                ########## Match the fields in Foursquare NYC dataset ##########
                # uid, pid, _, _, _, _, _, tim_orig = line.strip('\r\n').split('\t')

                ########## Convert character month to number to match with the clean_tweets_sample.txt ########## 
                # day, mon, date, time, zero, year = tim_orig.strip('\r\n').split(' ')
                # mon_num = str(strptime(mon, '%b').tm_mon)
                # tim = (year + '-' + mon_num + '-' + date + ' ' + time)
                
                if uid not in self.data:
                    self.data[uid] = [[pid, tim]]
                else:
                    self.data[uid].append([pid, tim])
                if pid not in self.venues:
                    self.venues[pid] = 1
                else:
                    self.venues[pid] += 1
        # print("The length of uid is: {}".format(len(self.data[uid])))
        # print("Count: {}".format(Counter(self.data)))
        cnt = len(self.data.keys())
        # pdb.set_trace()

        print(self.data.keys())        

        print("Printing the time of uid 14324035")
        for x in self.data['14324035']:
            print(x[1])
       
        print("Printing the time of uid 14604426")
        for x in self.data['14604426']:
            print(x[1])

        print("Number of unique UID in the dataset: {}".format(cnt))


    # ########### 2.0 basically filter users based on visit length and other statistics
    def filter_users_by_length(self):
        """
        [ expression for item in list if conditional ]
        """
        # filter out the uids with a number of visits that is larger than trace_len_min 
        uid_filtered = [x for x in self.data if len(self.data[x]) > self.trace_len_min]
        print("Number of UID with longer than 10 visits: {}".format(len(uid_filtered)))
        # Pack the uid and the number of visits together and sorted by the number of visits from high to low
        uid_filtered_sorted = sorted([(x, len(self.data[x])) for x in uid_filtered], key=lambda x: x[1], reverse=True)

        # filter out the pid with a number of visits that is larger than location_global 
        pid_filtered = [x for x in self.venues if self.venues[x] > self.location_global_visit_min]
        # Pack the pid and the number of visits together and sorted by the number of visits from high to low
        pid_filtered_sorted = sorted([(x, self.venues[x]) for x in pid_filtered], key=lambda x: x[1], reverse=True)

        pid_filtered = dict(pid_filtered_sorted)
        
        session_len_list = []

        for u in uid_filtered_sorted:
            uid = u[0]
            pid_n_time = self.data[uid]
            # Report the frequency of each element in the dictionary
            # topk contains all the places with number of visits
            # x[0] is pid in pid_n_time
            topk = Counter([x[0] for x in pid_n_time]).most_common()

            # topk1 is the locations visited more than once
            topk1 = [x[0] for x in topk if x[1] > 1]
            
            sessions = {}
            # One entry of pid_n_time: ['4c7c071181bca09382ee0415', '2012-4-16 20:15:54']
            for i, pid_n_time_enu in enumerate(pid_n_time):
                poi, tmd = pid_n_time_enu

                try:
                    # mktime: convert time the seconds since epoch in local time
                    tid = int(time.mktime(time.strptime(tmd, "%Y-%m-%d %H:%M:%S")))
                except Exception as e:
                    print('error:{}'.format(e))
                    continue
                
                # Use the length of the senssion as session id
                sid = len(sessions)

                # print(sid)
                # print("sid: {}".format(sid))
                if poi not in pid_filtered and poi not in topk1:
                    # if poi not in topk1:
                    continue
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [pid_n_time_enu]
                else:
                    if (tid - last_tid) / 3600 > self.hour_gap or len(sessions[sid - 1]) > self.session_max:
                        sessions[sid] = [pid_n_time_enu]
                    elif (tid - last_tid) / 60 > self.min_gap:
                        sessions[sid - 1].append(pid_n_time_enu)
                    else:
                        pass
                last_tid = tid
            sessions_filter = {}
            for s in sessions:
                # pdb.set_trace()
                if len(sessions[s]) >= self.filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s]))
            if len(sessions_filter) >= self.sessions_count_min:
                self.data_filter[uid] = {'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                                         'sessions': sessions_filter, 'raw_sessions': sessions}

        self.user_filter3 = [x for x in self.data_filter if
                             self.data_filter[x]['sessions_count'] >= self.sessions_count_min]
        print("user_filter3: {}".format(self.user_filter3))

    # ########### 3. build dictionary for users and location
    def build_users_locations_dict(self):
        for u in self.user_filter3:
            sessions = self.data_filter[u]['sessions']
            # pdb.set_trace()
            if u not in self.uid_list:
                self.uid_list[u] = [len(self.uid_list), len(sessions)]
            for sid in sessions:
                poi = [p[0] for p in sessions[sid]]
                for p in poi:
                    if p not in self.vid_list:
                        self.vid_list_lookup[len(self.vid_list)] = p
                        self.vid_list[p] = [len(self.vid_list), 1]
                    else:
                        self.vid_list[p][1] += 1

    # support for radius of gyration
    def load_venues(self):
        # with open(self.TWITTER_PATH, 'r') as fid:
        with open(self.DATASET_PATH + self.DATASET_NAME, 'r') as fid:
            for line in fid:
                _, uid, lon, lat, tim, _, _, tweet, pid = line.strip('\r\n').split('')
                # uid, pid, _, _, lat, lon, _, tim_orig = line.strip('\r\n').split('\t')
                
                # day, mon, date, time, zero, year = tim_orig.strip('\r\n').split(' ')
                # mon_num = str(strptime(mon, '%b').tm_mon)
                # tim = (year + '-' + mon_num + '-' + date + ' ' + time)
                self.pid_loc_lat[pid] = [float(lon), float(lat)]

    def venues_lookup(self):
        for vid in self.vid_list_lookup:
            pid = self.vid_list_lookup[vid]
            lon_lat = self.pid_loc_lat[pid]
            self.vid_lookup[vid] = lon_lat

    # ########## 4.0 prepare training data for neural network
    @staticmethod
    def tid_list(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        tid = tm.tm_wday * 24 + tm.tm_hour
        return tid

    @staticmethod
    def tid_list_48(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        if tm.tm_wday in [0, 1, 2, 3, 4]:
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return tid

    def prepare_neural_data(self):
        for u in self.uid_list:
            sessions = self.data_filter[u]['sessions']
            sessions_tran = {}
            sessions_id = []
            for sid in sessions:
                sessions_tran[sid] = [[self.vid_list[p[0]][0], self.tid_list_48(p[1])] for p in
                                      sessions[sid]]
                sessions_id.append(sid)
            split_id = int(np.floor(self.train_split * len(sessions_id)))
            train_id = sessions_id[:split_id]
            test_id = sessions_id[split_id:]
            pred_len = sum([len(sessions_tran[i]) - 1 for i in train_id])
            # print("pred_len: {}".format(pred_len))
            valid_len = sum([len(sessions_tran[i]) - 1 for i in test_id])
            # print("valid_len: {}".format(valid_len))
            train_loc = {}
            for i in train_id:
                for sess in sessions_tran[i]:
                    if sess[0] in train_loc:
                        train_loc[sess[0]] += 1
                    else:
                        train_loc[sess[0]] = 1
            # calculate entropy
            entropy = entropy_spatial(sessions)

            # calculate location ratio
            train_location = []
            for i in train_id:
                train_location.extend([s[0] for s in sessions[i]])
            train_location_set = set(train_location)
            test_location = []
            for i in test_id:
                test_location.extend([s[0] for s in sessions[i]])
            test_location_set = set(test_location)
            whole_location = train_location_set | test_location_set
            test_unique = whole_location - train_location_set
            location_ratio = len(test_unique) / len(whole_location)

            # calculate radius of gyration
            lon_lat = []
            for pid in train_location:
                try:
                    lon_lat.append(self.pid_loc_lat[pid])
                except:
                    print(pid)
                    print('error')
            lon_lat = np.array(lon_lat)
            center = np.mean(lon_lat, axis=0, keepdims=True)
            center = np.repeat(center, axis=0, repeats=len(lon_lat))
            rg = np.sqrt(np.mean(np.sum((lon_lat - center) ** 2, axis=1, keepdims=True), axis=0))[0]

            self.data_neural[self.uid_list[u][0]] = {'sessions': sessions_tran, 'train': train_id, 'test': test_id,
                                                     'pred_len': pred_len, 'valid_len': valid_len,
                                                     'train_loc': train_loc, 'explore': location_ratio,
                                                     'entropy': entropy, 'rg': rg}

            
    # ############# 6. save variables
    def get_parameters(self):
        parameters = {}
        parameters['trace_len_min'] = self.trace_len_min
        parameters['location_global_visit_min'] = self.location_global_visit_min
        parameters['hour_gap'] = self.hour_gap
        parameters['min_gap'] = self.min_gap
        parameters['session_max'] = self.session_max
        parameters['filter_short_session'] = self.filter_short_session
        parameters['sessions_min'] = self.sessions_count_min
        parameters['train_split'] = self.train_split

        return parameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_min', type=int, default=10, help="raw trace length filter threshold")
    parser.add_argument('--global_visit', type=int, default=10, help="location global visit threshold")
    parser.add_argument('--hour_gap', type=int, default=72, help="maximum interval of two trajectory points")
    parser.add_argument('--min_gap', type=int, default=10, help="minimum interval of two trajectory points")
    parser.add_argument('--session_max', type=int, default=10, help="control the length of session not too long")
    parser.add_argument('--session_min', type=int, default=5, help="control the length of session not too short")
    parser.add_argument('--sessions_min', type=int, default=5, help="the minimum amount of the good user's sessions")
    parser.add_argument('--train_split', type=float, default=0.8, help="train/test ratio")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_generator = DataFoursquare(trace_min=args.trace_min, global_visit=args.global_visit,
                                    hour_gap=args.hour_gap, min_gap=args.min_gap,
                                    session_min=args.session_min, session_max=args.session_max,
                                    sessions_min=args.sessions_min, train_split=args.train_split)

    parameters = data_generator.get_parameters()
    print('#'*10 + ' PARAMETER SETTINGS ' + '#'*10)
    print('\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
    print(10*'#' + ' START PROCESSING ' + 10*'#')
    print('1. load trajectory from {}'.format(data_generator.DATASET_PATH + data_generator.DATASET_NAME))
    data_generator.load_trajectory_from_tweets()
    print('2. filter users')
    data_generator.filter_users_by_length()
    print('3. build users/locations dictionary')
    data_generator.build_users_locations_dict()
    data_generator.load_venues()
    data_generator.venues_lookup()
    print('4. prepare data for neural network')
    data_generator.prepare_neural_data()
