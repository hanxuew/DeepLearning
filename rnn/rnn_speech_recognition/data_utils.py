import os
import numpy as np
import scipy.io.wavfile as sio
import python_speech_features
import random
import params
data_path = '../../data/speech_data/'
def parse_data(data_path, sentence_dict):
    path = []
    for i in sentence_dict.keys():
        relative_path = os.listdir(os.path.join(data_path, str(i)))
        absolute_path = list(map(lambda x: os.path.join(data_path, i , x), relative_path))
        path.extend(absolute_path)
    random.shuffle(path)
    return path

def get_speech(paths, sentence_dict):
    x = []
    y = []
    seq_length = []
    for path in paths:
        content = sio.read(path)
        sample_rate = content[0]
        mfcc = python_speech_features.mfcc(content[1], sample_rate)#获取音频的mfcc特征
        key = path.split('/')[-1].split('\\')[0]
        x.append(mfcc)
        y.append(sentence_dict[key])
        seq_length.append(len(mfcc))
    return x, y, seq_length #100帧是1s seq_length对应音频的帧数

def get_batch(batchsize, x, y, seq_length):
    pad = [0. for _ in range(len(x[0][0]))]
    for i in range(len(x)//batchsize):
        t_x = x[i * batchsize: (i + 1) * batchsize]
        t_y = x[i * batchsize: (i + 1) * batchsize]
        t_seq = seq_length[-batchsize:]
        max_l = max(t_seq) #获取每个batch中最长序列
        res_x = []
        for mfcc in t_x:
            mfcc = np.concatenate((mfcc, np.tile(pad, (max_l - len(mfcc), 1))), axis=0)  # 以pad的形式补充成等长的帧数
            res_x.append(mfcc)
        yield res_x, t_y, t_seq
#
    remain = np.shape(x)[0] % batchsize
    if remain != 0:
        t_x = x[-remain:]
        t_y = y[-remain:]
        t_seq = seq_length[-batchsize:]
        max_l = max(t_seq)  # 获取每个batch中最长序列
        res_x = []
        for mfcc in t_x:
            mfcc = np.concatenate((mfcc, np.tile(pad, (max_l - len(mfcc), 1))), axis=0)  # 以pad的形式补充成等长的帧数
            res_x.append(mfcc)
        yield res_x, t_y, t_seq


if __name__ == '__main__':
    sentence_dict = {'yes' :1, 'no': 0}
    data = parse_data(data_path, sentence_dict)
    data_speech = get_speech(data, sentence_dict)
    # data = np.array(data_speech[0])
    # print(data.shape)#4752 99 13