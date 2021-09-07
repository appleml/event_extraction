import os
import copy
from data_prepare.file import biofile, sentence, sent_genia, sent_parse
import data_prepare.delete_nostandard as dns
import data_prepare.data_utils as util
from collections import Counter

'''
读原始数据，读parser解析出的数据
(1)删除掉四个类型的trigger(2)删除重复事件(4)删除跨句子事件
(3)保留多词组成trigger引发的事件, 这是与gcn_input.py的不同之处
data_path 是一个句子的语料,用于识别trigger和join protein
data_two_path的目的是为了识别trigger与argument的关系以及binding argument
注意 protein 使用的标签系统是BIO, trigger使用的标签系统也是BIO(2020年11月09日)
'''
def read_data(data_path, genia_parse_path):
    file_set = []
    files = os.listdir(data_path)
    all_event_num = 0
    invalid_event_num = 0
    valid_event_num = 0

    for file_name in files:
        fileinfo = biofile(file_name)
        with open(data_path + "/" + file_name, "r") as data_fp, open(genia_parse_path + "/" + file_name, "r") as genia_parse_fp:
            old_sent = ""
            str_prot_set = []
            str_prot_idxs = []
            str_jprot_set = []
            str_trig_idxs = []
            str_trig_set = []
            str_event_set = []
            str_event_idx = []

            for line in data_fp:
                line = line.strip()
                if not len(line):

                    # 处理genia_data中的数据
                    gword_set = []
                    gpos_set = []
                    gprot_set = []
                    gjprot_set = []
                    cjprot_set = []
                    ctrig_set = []
                    trigtype_set = []

                    # parse数据
                    ppos_set = []
                    head_set = []
                    deptype_set = []

                    genia_parse_line = genia_parse_fp.readline()
                    genia_parse_line = genia_parse_line.strip().strip('\n')
                    while len(genia_parse_line) != 0:
                        genia_parse = genia_parse_line.split("\t###\t")
                        # genia_line信息处理，为识别trigger做准备
                        # (word+"\t"+pos+"\t"+fea1+"\t"+fea2+"\t"+prot+"\t"+cjprot+"\t"+ctrig+"\t"+jprot+"\t"+trigtype+"\n")
                        genia_line = genia_parse[0]
                        genia_info = genia_line.split("\t")
                        gword_set.append(genia_info[0])
                        gpos_set.append(genia_info[1])
                        gprot_set.append(genia_info[4])
                        cjprot_set.append(genia_info[5])
                        ctrig_set.append(genia_info[6])
                        gjprot_set.append(genia_info[7])
                        if genia_info[8] == "Entity":
                            trigtype_set.append('Other')
                        else:
                            trigtype_set.append(genia_info[8])

                        # parse信息处理
                        # 1	Down-regulation	_	NN	_	_	0	root	_	_
                        parse_line = genia_parse[1]
                        parse_info = parse_line.split("\t")
                        ppos_set.append(parse_info[3])
                        head_set.append(parse_info[6])
                        deptype_set.append(parse_info[7])

                        genia_parse_line = genia_parse_fp.readline()
                        genia_parse_line = genia_parse_line.strip().strip('\n')

                    new_sent = " ".join(gword_set)
                    assert old_sent.replace(" ", "") == new_sent.replace(" ", "")

                    # --------(1)删除掉四个类型的trigger(2)删除重复事件(3)删除多词trigger引发的事件(4)删除跨句子事件--------------
                    valid_event_set = dns.delete_invalid_event_two(str_event_set, str_prot_idxs, str_trig_idxs, str_event_idx)
                    invalid_event_num += len(str_event_set) - len(valid_event_set)
                    valid_event_num += len(valid_event_set)
                    # 将str_prot, str_trig, str_event转换成event
                    prot_dict = util.prot_str2entity(old_sent, new_sent, str_prot_set)
                    jprot_dict = util.prot_str2entity(old_sent, new_sent, str_jprot_set)
                    trig_dict = util.trig_str2entity(old_sent, new_sent, str_trig_set)

                    event_dict, event_set = util.event_str2entity(file_name, valid_event_set, trig_dict, prot_dict)
                    assert len(valid_event_set) == len(event_dict)

                    sentence_genia = sent_genia(copy.deepcopy(gword_set), copy.deepcopy(gpos_set), copy.deepcopy(gprot_set))
                    sentence_genia.jprot_set = copy.deepcopy(gjprot_set)
                    sentence_genia.trig_types = copy.deepcopy(trigtype_set)
                    sentence_genia.cjprot_set = copy.deepcopy(cjprot_set)
                    sentence_genia.ctrig_set = copy.deepcopy(ctrig_set)

                    sentence_parse = sent_parse(copy.deepcopy(gword_set), copy.deepcopy(ppos_set), copy.deepcopy(head_set), copy.deepcopy(deptype_set))
                    sent_info = sentence(old_sent, new_sent, copy.deepcopy(prot_dict), sentence_genia, sentence_parse)
                    sent_info.jprot_dict = copy.deepcopy(jprot_dict)
                    sent_info.trig_dict = copy.deepcopy(trig_dict)
                    sent_info.event_dict = copy.deepcopy(event_dict)
                    sent_info.event_set = copy.deepcopy(event_set)

                    fileinfo.add_sent(sent_info)

                    old_sent = ""
                    str_prot_set.clear()
                    str_jprot_set.clear()
                    str_trig_set.clear()
                    str_event_set.clear()
                    str_event_idx.clear()
                    str_prot_idxs.clear()
                    str_trig_idxs.clear()

                elif line.startswith("# "):
                    str_prot_set.append(line)
                    prot_info = line.split()
                    str_prot_idxs.append(prot_info[1])

                elif line.startswith("$"):
                    str_jprot_set.append(line)

                elif line.startswith("@"):# @ T20 Regulation S2 160 172 306 318 deregulation
                    trig_info = line.split()
                    if trig_info[2] not in ["Entity", 'Anaphora']:
                        str_trig_set.append(line)
                    else:
                        assert trig_info[2] in ["Entity", 'Anaphora']
                        str_trig_idxs.append(trig_info[1])

                elif line.startswith("%"): # % E3 Regulation:T21 Theme:T4
                    all_event_num += 1
                    str_event, event_idx = util.process_strevent(line)
                    if str_event not in str_event_set: ## 删除重复event
                        str_event_set.append(str_event)
                        str_event_idx.append(event_idx)
                    # if str_event.split(" ")[2].split(":")[0] in ['Protein_modification', 'Phosphorylation', 'Ubiquitination',  'Acetylation', 'Deacetylation']:
                    #     print(str_event)
                else:
                    old_sent = line

        file_set.append(fileinfo)
    comp_prev_next_sent(file_set)
    print("所有事件个数:", all_event_num)
    print("跨句子事件个数：", invalid_event_num)
    print("有效事件个数：", valid_event_num)
    print("#################################")
    return file_set

## 补全当前句子的前后句
def comp_prev_next_sent(fileinfo):
    for file_entity in fileinfo:
        sent_set = file_entity.sent_set
        for sent_ids in range(len(sent_set)):
            if sent_ids == 0 and len(sent_set) > 1:
                next_sent_entity = sent_set[sent_ids+1]
                next_sentence = next_sent_entity.new_sent
                curr_sent_entity = sent_set[sent_ids]
                curr_sent_entity.next_sentence = next_sentence

            elif sent_ids == len(sent_set)-1:
                prev_sent_entity = sent_set[sent_ids - 1]
                prev_sentence = prev_sent_entity.new_sent
                curr_sent_entity = sent_set[sent_ids]
                curr_sent_entity.prev_sentence = prev_sentence

            else:
                prev_sent_entity = sent_set[sent_ids-1]
                next_sent_entity = sent_set[sent_ids+1]
                prev_sentence = prev_sent_entity.new_sent
                next_sentence = next_sent_entity.new_sent

                curr_sent_entity = sent_set[sent_ids]
                curr_sent_entity.prev_sentence = prev_sentence
                curr_sent_entity.next_sentence = next_sentence

'''
读取test集中数据
'''
def read_test_data(data_path, parse_path):
    processed_files = []
    files = os.listdir(data_path)
    for file_name in files:
        fileinfo = biofile(file_name)
        with open(data_path + "/" + file_name, "r") as data_fp, open(parse_path + "/" + file_name, "r") as parse_fp:
            old_sent = ""
            str_prot_set = list()

            for line in data_fp:
                line = line.strip()
                if not len(line):
                    ## genia 信息数据
                    gword_set = list()
                    gpos_set = list()
                    gprot_set = list()

                    # 处理parse_data中的数据
                    ppos_set = []
                    head_set = []
                    deptype_set = []

                    genia_parse_line = parse_fp.readline()
                    genia_parse_line = genia_parse_line.strip().strip('\n')
                    while len(genia_parse_line) != 0:
                        genia_parse = genia_parse_line.split("\t###\t")
                        # genia_line: Down-regulation	NN	B-NP	O	Other
                        # (word+"\t"+pos+"\t"+fea1+"\t"+fea2+"\t"+prot")
                        genia_line = genia_parse[0].strip()
                        genia_info = genia_line.split("\t")
                        gword_set.append(genia_info[0])
                        gpos_set.append(genia_info[1])
                        gprot_set.append(genia_info[4])

                        # parse信息处理
                        # 1	Down-regulation	Down-regulation	NN	NN	_	0	root	_	_
                        parse_line = genia_parse[1].strip()
                        parse_info = parse_line.split("\t")
                        ppos_set.append(parse_info[3])
                        head_set.append(parse_info[6])
                        deptype_set.append(parse_info[7])

                        genia_parse_line = parse_fp.readline()
                        genia_parse_line = genia_parse_line.strip().strip('\n')

                    new_sent = " ".join(gword_set)
                    if old_sent.replace(" ", "") != new_sent.replace(" ", ""):
                        print("#########################################")
                        print(old_sent.replace(" ", ""))
                        print(new_sent.replace(" ", ""))
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                    assert old_sent.replace(" ", "") == new_sent.replace(" ", "")

                    prot_dict = util.prot_str2entity(old_sent, new_sent, str_prot_set)

                    sentence_genia = sent_genia(copy.deepcopy(gword_set), copy.deepcopy(gpos_set), copy.deepcopy(gprot_set))
                    sentence_parse = sent_parse(copy.deepcopy(gword_set), copy.deepcopy(ppos_set), copy.deepcopy(head_set), copy.deepcopy(deptype_set))

                    sent_info = sentence(old_sent, new_sent, copy.deepcopy(prot_dict), sentence_genia, sentence_parse)
                    fileinfo.add_sent(sent_info)

                    old_sent = ""
                    str_prot_set.clear()

                elif line.startswith("#"):
                    str_prot_set.append(line)

                elif line.startswith("$") or line.startswith("@") or line.startswith("%"):
                    continue
                else:
                    old_sent = line

        processed_files.append(fileinfo)
    comp_prev_next_sent(processed_files)
    return processed_files

'''
统计关系类型的个数
Theme类型的个数
Cause类型的个数
嵌套事件的个数
% E2 Gene_expression:T19 Theme:T1
'''
def statistics_info(str_event_set):
    num_nested_event = 0
    num_theme = 0
    num_cause = 0
    for event in str_event_set:
        event_info = event.split()
        fargu_info = event_info[3].split(":")
        fargu_type = fargu_info[0]
        fargu_id = fargu_info[1]
        if fargu_type == "Theme":
            num_theme += 1
        elif fargu_type == "Cause":
            num_cause += 1

        if fargu_id.startswith("E"):
            num_nested_event += 1
        elif len(event_info) > 4:
            sargu_info = event_info[4].split(":")
            sargu_type = sargu_info[0]
            sargu_id = sargu_info[1]
            if sargu_type.startswith("Theme"):
                num_theme += 1
            elif sargu_type.startswith("Cause"):
                num_cause += 1
            if sargu_id.startswith("E"):
                num_nested_event += 1
    return num_nested_event, num_theme, num_cause
