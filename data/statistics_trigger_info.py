import os
import copy
from data_prepare.file import biofile, sentence, sent_genia, sent_parse
import data_prepare.delete_nostandard as dns
import data_prepare.data_utils as util

'''
读原始数据，读parser解析出的数据
(1)删除掉四个类型的trigger(2)删除重复事件(3)删除多词trigger引发的事件(4)删除跨句子事件
data_path 是一个句子的语料,用于识别trigger和join protein
data_two_path的目的是为了识别trigger与argument的关系以及binding argument
'''
def read_data(data_path):
    files = os.listdir(data_path)
    one_wordtrig_number = 0
    multi_wordtrig_number = 0

    for file_name in files:
        with open(data_path + "/" + file_name, "r") as data_fp:
            for line in data_fp:
                if line.startswith("@"): # @ T20 Regulation S2 160 172 306 318 deregulation
                    trig_info = line.split()
                    if trig_info[2] not in ["Entity", 'Anaphora', 'Ubiquitination', 'Protein_modification']:
                        if len(trig_info) == 9:
                            one_wordtrig_number += 1
                        elif len(trig_info) > 9:
                            multi_wordtrig_number += 1
                        else:
                            print('没有其他情况了')
    print("一个词的trigger一共有:", one_wordtrig_number)
    print("多个词的trigger一共有:", multi_wordtrig_number)

if __name__ == '__main__':
    train_data = "/home/fang/fangcode/gcn-deprela-bilstm-protrig-gcn-relabind/dataset/2011_data_one/2011_train"
    read_data(train_data)
    devel_data = "/home/fang/fangcode/gcn-deprela-bilstm-protrig-gcn-relabind/dataset/2011_data_one/2011_devel"
    read_data(devel_data)