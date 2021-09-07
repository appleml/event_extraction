## 该模块中的函数都是为 trainer.py 服务的
from utils import constant, torch_utils
import torch
## file_name, words, prot_dict, char_ids, genia_info, parse_info, entity_pairs, pair_labels, gold_events
def unpack_batch(batch, cuda):
    file_name, words, char_ids, prev_sent, next_sent, prot_dict, genia_info, parse_info, entity_pairs, bind_argus, pmode_argus, regu_argus, gold_events = batch
    genia_words, pos_ids, prot_ids, cjprot_ids, ctrig_ids, prot_labels, jprot_labels, jprot_ids, trig_labels, tlabel_ids = genia_info
    parse_posIds, deprel, head = parse_info
    if cuda:
        char_ids = char_ids.cuda()
        genia_feas = [gf.cuda() for gf in genia_info[1:5]]
        parse_posIds = parse_posIds.cuda()
        deprel = deprel.cuda()
        head = head.cuda()
    else:
        genia_feas = [gf for gf in genia_info[1:5]]

    file_name = batch[0]
    words = batch[1]
    genia_input = (words, genia_feas[0], genia_feas[1], genia_feas[2], genia_feas[3], prot_labels, jprot_labels, jprot_ids, trig_labels, tlabel_ids)
    parse_input = (parse_posIds, deprel, head)

    return file_name, words, char_ids, prev_sent, next_sent, prot_dict, genia_input, parse_input, entity_pairs, bind_argus, pmode_argus, regu_argus, gold_events

def unpack_devbatch(batch, prev_jprot_set, prev_ctrig_dict, cuda):
    file_name, words, char_ids, prev_sent, next_sent, prot_dict, genia_info, parse_info, entity_pairs, bind_argus, pmode_argus, regu_argus, gold_events = batch
    genia_words, pos_ids, prot_ids, cjprot_ids, ctrig_ids, prot_labels, jprot_labels, jprot_ids, trig_labels, tlabel_ids = genia_info
    parse_posIds, deprel, head = parse_info

    pred_cjprot_seq = get_context_jport_BIO(words, prot_dict, prev_jprot_set)
    pred_ctrig_seq = get_context_trig_BIO(words, prev_ctrig_dict)
    pred_genia_jprot_ids = map_to_ids(pred_cjprot_seq, constant.JPROT_TO_ID)
    pred_genia_ctrig_ids = map_to_ids(pred_ctrig_seq, constant.TRIGGER_TO_ID)
    pred_genia_jprot_ids = get_long_tensor([pred_genia_jprot_ids], 1)
    pred_genia_ctrig_ids = get_long_tensor([pred_genia_ctrig_ids], 1)

    if cuda:
        char_ids = char_ids.cuda()
        genia_feas = [gf.cuda() for gf in genia_info[1:5]]
        pred_genia_jprot_ids = pred_genia_jprot_ids.cuda()
        pred_genia_ctrig_ids = pred_genia_ctrig_ids.cuda()

        parse_posIds = parse_posIds.cuda()
        deprel = deprel.cuda()
        head = head.cuda()
    else:
        genia_feas = [gf for gf in genia_info[1:5]]

    file_name = batch[0]
    words = batch[1]
    genia_input = (words, genia_feas[0], genia_feas[1], pred_genia_jprot_ids, pred_genia_ctrig_ids, prot_labels, jprot_labels, jprot_ids,
    trig_labels, tlabel_ids)
    parse_input = (parse_posIds, deprel, head)

    return file_name, words, char_ids, prev_sent, next_sent, prot_dict, genia_input, parse_input, entity_pairs, bind_argus, pmode_argus, regu_argus, gold_events


## 处理 test 语料
# file_name, old_sent, words, char_ids, prot_dict, genia_info, parse_info
def unpack_testbatch(batch, prev_jprot_set, prev_ctrig_dict, cuda):
    file_name, old_sent, words, char_ids, prev_sent, next_sent, prot_dict, genia_info, parse_info = batch
    genia_words, pos_ids, prot_ids, prot_labels = genia_info
    parse_posIds, deprel, head = parse_info
    ## 对前面句子中识别出的protein, trigger处理成序列
    pred_genia_jprot_ids, pred_genia_ctrig_ids = process_jprot_trig_seq(words, prot_dict, prev_jprot_set, prev_ctrig_dict)

    if cuda:
        char_ids = char_ids.cuda()
        genia_feas = [gf.cuda() for gf in genia_info[1:3]]
        pred_genia_jprot_ids = pred_genia_jprot_ids.cuda()
        pred_genia_ctrig_ids = pred_genia_ctrig_ids.cuda()

        parse_posIds = parse_posIds.cuda()
        deprel = deprel.cuda()
        head = head.cuda()
    else:
        genia_feas = [gf for gf in genia_info[1:3]]

    genia_input = (words, genia_feas[0], genia_feas[1], pred_genia_jprot_ids, pred_genia_ctrig_ids, prot_labels)
    parse_input = (parse_posIds, deprel, head)

    return file_name[0], old_sent, words, char_ids, prev_sent, next_sent, prot_dict, genia_input, parse_input

def process_jprot_trig_seq(words, prot_dict, prev_jprot_set, prev_ctrig_dict):
    # 前几个句子中识别出的蛋白质
    pred_cjprot_seq = get_context_jport_bio(words, prot_dict, prev_jprot_set)
    pred_genia_jprot_ids = map_to_ids(pred_cjprot_seq, constant.JPROT_TO_ID)
    pred_genia_jprot_ids = get_long_tensor([pred_genia_jprot_ids], 1)  # batch_size = 1

    # 前面几个句子中识别出的trigger
    pred_ctrig_seq = get_context_trig_bio(words, prev_ctrig_dict)
    pred_genia_ctrig_ids = map_to_ids(pred_ctrig_seq, constant.TRIGGER_TO_ID)
    pred_genia_ctrig_ids = get_long_tensor([pred_genia_ctrig_ids], 1)

    return pred_genia_jprot_ids, pred_genia_ctrig_ids

'''
predict中生成当前句子的 context join protein特征
sent_prots是句子中标注的蛋白质集合
prot_seq是标注的蛋白质序列
context_jprots是前面句子中出现的join protein
'''
def get_context_jport_bio(word_set, prot_dict, prev_jprot_set):
    sent_len = len(word_set[0].split())
    cjprot_seqs = ["Other"] * sent_len

    for _, prot in prot_dict.items():
        prot_name = prot.prot_name
        prot_start = prot.prot_start
        prot_end = prot.prot_end
        if prot_name in prev_jprot_set or prot_name.lower() in prev_jprot_set:
            if prot_start == prot_end:
                cjprot_seqs[prot_start] = "B-Protein"
            else:
                for i in range(prot_start, prot_end + 1):
                    if i == prot_start:
                        cjprot_seqs[i] = "B-Protein"
                    else:
                        cjprot_seqs[i] = "I-Protein"
    return cjprot_seqs


'''
 生成 previous sentence trigger feature
 context_trig可能不会大, 因为篇章不是太长
 该函数需要调试
'''
def get_context_trig_bio(word_set, context_triginfo_dict):
    sent = word_set[0]
    sent_len = len(sent.split())
    ctrig_seqs = ["Other"] * sent_len
    trig_info_set = sorted(context_triginfo_dict.items(), key=lambda item: len(item[0].split()), reverse=True)
    for trig_info in trig_info_set:
        maybe_trig_name, maybe_trig_type = trig_info[0], trig_info[1]
        trig_num = len(maybe_trig_name.split())
        begin = 0
        while sent.find(maybe_trig_name, begin) != -1 or sent.find(maybe_trig_name.lower(), begin) != -1:
            char_start = sent.find(maybe_trig_name, begin)
            if char_start == -1:
                char_start = sent.find(maybe_trig_name.lower(), begin)

            word_start = len(sent[0:char_start].split())
            if ctrig_seqs[word_start] == "Other":
                for i in range(word_start, word_start+trig_num):
                    if i == word_start:
                        ctrig_seqs[i] = "B-"+maybe_trig_type
                    else:
                        ctrig_seqs[i] = "I-"+maybe_trig_type
            begin = char_start + len("you")
    return ctrig_seqs

'''
判断trigger是否与argu有关系
trig_argu_pair: event_idx_set, trig, trig_type_idx, trig_posit, argu_trig, argu_type_idx, argu_posit
train:(event_idx_set, new_word, trig, argu, trig_posit, argu_posit)
devel:(trig, trig_type_idx, trig_posit, prot, argu_type_idx, argu_posit)
适用情况: 一个句子有k个实体对需要判断关系, 则批处理这k个实体对
batch_entity_pairs_first: trig_type_idx, trig_posit, argu_type_idx, argu_posit
batch_entity_pairs_second: event_idx_set, new_words, trig, argu_trig, new_trig_posit, new_argu_posit
'''
def process_batch_pair(entity_pairs_info, cuda, mode):
    if mode == "train":
        batch_entity_pairs_first, batch_entity_pairs_second, pair_labels, batch_entity_exams = entity_pairs_info

    elif mode == "devel" or mode == "test":
        batch_entity_pairs_first, batch_entity_pairs_second = entity_pairs_info

    pair_size = len(batch_entity_pairs_second)
    batch_size = 4
    first_input_data = [batch_entity_pairs_first[i:i+batch_size] for i in range(0, pair_size, batch_size)]
    second_input_data = [batch_entity_pairs_second[i:i + batch_size] for i in range(0, pair_size, batch_size)]
    if mode == "train":
        rela_label = [constant.RELA_TO_ID[rela_type] for rela_type in pair_labels]
        input_label = [rela_label[i:i + batch_size] for i in range(0, pair_size, batch_size)]

    first_input_set = []
    second_input_set = []
    label_ids = []
    for j_th, mini_second_batch in enumerate(second_input_data):
        batch_size = len(mini_second_batch)
        ## 第一个编码器的输入
        mini_first_batch = first_input_data[j_th]
        mini_first_batch = list(zip(*mini_first_batch))

        trig_type_id = get_long_tensor(mini_first_batch[0], batch_size)
        first_trig_posit = get_long_tensor(mini_first_batch[1], batch_size)
        first_trig_mask = first_trig_posit.eq(0).eq(0).unsqueeze(2).bool()

        argu_type_id = get_long_tensor(mini_first_batch[2], batch_size)
        first_argu_posit = get_long_tensor(mini_first_batch[3], batch_size)
        first_argu_mask = first_argu_posit.eq(0).eq(0).unsqueeze(2).bool()

        ## 第二个编码器的输入
        mini_second_batch = list(zip(*mini_second_batch))
        new_words = mini_second_batch[0]

        second_trig_posit = get_long_tensor(mini_second_batch[1], batch_size) ## 因为batch_size的原因不能在放在处理数据那里
        second_trig_mask = second_trig_posit.eq(0).eq(0).unsqueeze(2).bool()

        second_argu_posit = get_long_tensor(mini_second_batch[2], batch_size)
        second_argu_mask = second_argu_posit.eq(0).eq(0).unsqueeze(2).bool()

        if cuda:
            trig_type_id = trig_type_id.cuda()
            first_trig_mask = first_trig_mask.cuda()
            argu_type_id = argu_type_id.cuda()
            first_argu_mask = first_argu_mask.cuda()

            second_trig_mask = second_trig_mask.cuda()
            second_argu_mask = second_argu_mask.cuda()

        mini_batch_first_input = (trig_type_id, first_trig_mask, argu_type_id, first_argu_mask)
        mini_batch_second_input = (new_words, second_trig_mask, second_argu_mask)
        first_input_set.append(mini_batch_first_input)
        second_input_set.append(mini_batch_second_input)

        if mode == "train":
            rela_label = torch.LongTensor(input_label[j_th])
            if cuda:
                rela_label = rela_label.cuda()

            label_ids.append(rela_label)

    if mode == "train":
        return first_input_set, second_input_set, label_ids
    else:
        return first_input_set, second_input_set

'''
在 Binding event 中，判断两个论元是否在同一个事件中：
(new_words, trig_mask, argu1_mask, argu2_mask)
train: bind_pair_input, bind_pair_seq, cuda
devel: bind_pair_input, cuda

'''
def process_batch_br(params, cuda, mode):
    if mode == "train":
        inputs_first, inputs_second, pair_labels, pair_exams = params
    elif mode == "devel" or mode == "test":
        inputs_first, inputs_second = params

    batch_size = len(inputs_first)
    batch_first = list(zip(*inputs_first))

    argu1_type_id = get_long_tensor(batch_first[0], batch_size)
    argu1_posit_first = get_long_tensor(batch_first[1], batch_size)
    argu1_mask_first = argu1_posit_first.eq(0).eq(0).unsqueeze(2).bool()

    argu2_type_id = get_long_tensor(batch_first[2], batch_size)
    argu2_posit_first = get_long_tensor(batch_first[3], batch_size)
    argu2_mask_first = argu2_posit_first.eq(0).eq(0).unsqueeze(2).bool()
    ##############################################################33
    batch_size = len(inputs_second)
    batch = list(zip(*inputs_second))

    new_words = batch[0]
    argu1_posit_second = get_long_tensor(batch[1], batch_size)
    argu1_mask_second = argu1_posit_second.eq(0).eq(0).unsqueeze(2).bool()
    argu2_posit_second = get_long_tensor(batch[2], batch_size)
    argu2_mask_second = argu2_posit_second.eq(0).eq(0).unsqueeze(2).bool()

    if cuda:
        argu1_type_id = argu1_type_id.cuda()
        argu1_mask_first = argu1_mask_first.cuda()
        argu2_type_id = argu2_type_id.cuda()
        argu2_mask_first = argu2_mask_first.cuda()
        #########################################33
        second_argu1_mask = argu1_mask_second.cuda()
        second_argu2_mask = argu2_mask_second.cuda()

    br_inputs_first = ((argu1_type_id, argu1_mask_first, argu2_type_id, argu2_mask_first))
    br_inputs_second = (new_words, second_argu1_mask, second_argu2_mask)

    if mode == "train":
        bind_label = torch.LongTensor(pair_labels)
        if cuda:
            br_labels = bind_label.cuda()
        return br_inputs_first, br_inputs_second, br_labels
    else:
        return br_inputs_first, br_inputs_second


def process_batch_br2(params, cuda, mode):
    if mode == "train":
        inputs_first, inputs_second, pair_labels, pair_exams = params
    elif mode == "devel" or mode == "test":
        inputs_first, inputs_second = params

    pair_size = len(inputs_second)
    batch_size = 6
    first_input_data = [inputs_first[i:i+batch_size] for i in range(0, pair_size, batch_size)]
    second_input_data = [inputs_second[i:i+batch_size] for i in range(0, pair_size, batch_size)]
    if mode == "train":
        input_labels = [pair_labels[i:i + batch_size] for i in range(0, pair_size, batch_size)]

    first_input_set = []
    second_input_set = []
    label_ids_set = []
    for j_th, mini_second_batch in enumerate(second_input_data):
        batch_size = len(mini_second_batch)
        ## 第一个编码器的输入
        mini_first_batch = first_input_data[j_th]
        mini_first_batch = list(zip(*mini_first_batch))

        argu1_type_id = get_long_tensor(mini_first_batch[0], batch_size)
        argu1_posit_first = get_long_tensor(mini_first_batch[1], batch_size)
        argu1_mask_first = argu1_posit_first.eq(0).eq(0).unsqueeze(2).bool()

        argu2_type_id = get_long_tensor(mini_first_batch[2], batch_size)
        argu2_posit_first = get_long_tensor(mini_first_batch[3], batch_size)
        argu2_mask_first = argu2_posit_first.eq(0).eq(0).unsqueeze(2).bool()

        ## 第二个编码器的输入
        mini_second_batch = list(zip(*mini_second_batch))
        new_words = mini_second_batch[0]

        new_words = mini_second_batch[0]
        argu1_posit_second = get_long_tensor(mini_second_batch[1], batch_size)
        argu1_mask_second = argu1_posit_second.eq(0).eq(0).unsqueeze(2).bool()
        argu2_posit_second = get_long_tensor(mini_second_batch[2], batch_size)
        argu2_mask_second = argu2_posit_second.eq(0).eq(0).unsqueeze(2).bool()

        if cuda:
            argu1_type_id = argu1_type_id.cuda()
            argu1_mask_first = argu1_mask_first.cuda()
            argu2_type_id = argu2_type_id.cuda()
            argu2_mask_first = argu2_mask_first.cuda()
            #########################################33
            argu1_mask_second = argu1_mask_second.cuda()
            argu2_mask_second = argu2_mask_second.cuda()

        if mode == "train":
            rela_label_ids = torch.LongTensor(input_labels[j_th])
            if cuda:
                rela_label_ids = rela_label_ids.cuda()
            label_ids_set.append(rela_label_ids)

        mini_batch_first_input = (argu1_type_id, argu1_mask_first, argu2_type_id, argu2_mask_first)
        mini_batch_second_input = (new_words, argu1_mask_second, argu2_mask_second)
        first_input_set.append(mini_batch_first_input)
        second_input_set.append(mini_batch_second_input)


    if mode == "train":
        return first_input_set, second_input_set, label_ids_set

    else:
        return first_input_set, second_input_set

##------------------------------------------------------------
## loader.py 和 test_loader.py 共用的部分
def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + (batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]# sorted_all记录每条example在原来的位置

'''
为了char 进行CNN卷积特征
words: 是一个List
conv_filter_size: 卷积核大小
char2id: 
max_len: batch中最长的句子的长度
max_word_len: batch中最长的单词的长度
'''
def get_char_ids(word_seq, char2id):
    words = word_seq.split()
    max_word_len = max([len(token) for token in words])
    char_seq = list()

    for word in words:
        word_char_seq = list()
        for c in word[0:min(len(word), max_word_len)]:
            if c in char2id:
                word_char_seq.append(char2id[c])
            else:
                word_char_seq.append(char2id['<UNK>'])
        pad_len = max_word_len - len(word)
        for i in range(0, pad_len):
            word_char_seq.append(char2id['<PAD>'])
        char_seq.append(word_char_seq)
    return char_seq

#--------------------------------------------------------------
## 生成句子级别中trigger的index
## 用于在 trainer.py 中的predict, test没有用
def gen_trigidx(prot_dict):
    index_set = list()
    for prot_idx, prot in prot_dict.items():
        prot_num = int(prot_idx[1:])
        index_set.append(prot_num)
    max_idx = max(index_set)
    return max_idx

## 计算trigger的位置信息
#############################################################################
## 计算trigger的位置信息
'''
## 默认old_sent和new_sent是string
## 已经给出了protein在全文的地址
(self, trig_idx, trig_name, trig_type, trig_oldchar_start, trig_oldchar_end, trig_newchar_start, trig_newchar_end, trig_start, trig_end)
 将trigger (1) 相对句子中的位置映射为全文的position
          (2) 转换成string
trigger("T"+str(trig_idx), token, label, None, None, None, None, word_idx, word_idx)
'''
def trig_context_posit(file_name, old_sent, new_sent, prot_dict, trig_dict, context):
    new_sent_list = new_sent.split()
    _, first_prot = next(iter(prot_dict.items()))
    for trig_idx, trig_entity in trig_dict.items():
        trig_start = trig_entity.trig_start
        trig_end = trig_entity.trig_end
        trig_name = trig_entity.trig_name
        ## 先计算 trig_newchar_start, trig_newchar_end
        start_prefix = " ".join(new_sent_list[0:trig_start])
        if start_prefix == "":
            trig_entity.trig_newchar_start = len(start_prefix)
        else:
            trig_entity.trig_newchar_start = len(start_prefix)+1 ## 加1是空格
        end_prefix = " ".join(new_sent_list[0:trig_end+1])
        trig_entity.trig_newchar_end = len(end_prefix)
        assert new_sent[trig_entity.trig_newchar_start:trig_entity.trig_newchar_end] == trig_entity.trig_name

        ## 根据trig_newchar_start, trig_newchar_end， 然后计算 trig_oldchar_start， trig_oldchar_end
        trig_entity.trig_oldchar_start = comp_oldchar_position(old_sent, new_sent, trig_entity.trig_newchar_start, False)
        trig_entity.trig_oldchar_end = comp_oldchar_position(old_sent, new_sent, trig_entity.trig_newchar_end, True)
        #print(file_name, "--------------------------")
        #print(old_sent[trig_entity.trig_oldchar_start:trig_entity.trig_oldchar_end], "---", trig_entity.trig_name)
        assert old_sent[trig_entity.trig_oldchar_start:trig_entity.trig_oldchar_end].replace(" ", "") == trig_name.replace(" ", "")

        ## trig_oldchar_start， trig_oldchar_end，最后计算 trig_context_start, trig_context_end
        trigger_name = old_sent[trig_entity.trig_oldchar_start:trig_entity.trig_oldchar_end]
        trig_entity.trig_context_start, trig_entity.trig_context_end = comp_article_position(file_name, old_sent, context, trigger_name, trig_entity.trig_oldchar_start, first_prot)

# 根据在new_sent中的位置计算在old_sent中的位置
def comp_oldchar_position(old_sent, new_sent, newchar_position, is_end):
    oldchar_position = newchar_position

    #while len(newchar_prefix) != len(oldchar_prefix):
    while new_sent[0:newchar_position].replace(" ", "") != old_sent[0:oldchar_position].replace(" ",""):
        oldchar_position = oldchar_position-1

    if is_end == True and new_sent[0:newchar_position].replace(" ", "") == old_sent[0:oldchar_position-1].replace(" ",""):
        oldchar_position = oldchar_position-1

    assert old_sent[0:oldchar_position].replace(" ", "") == new_sent[0:newchar_position].replace(" ", "")
    return oldchar_position

def comp_article_position(file_name, old_sent, context, trig_name, trig_oldchar_start, first_prot):
    trig_context_start = -1
    trig_context_end = -1
    prot_context_start = first_prot.prot_context_start
    prot_context_end = first_prot.prot_context_end

    assert prot_context_start != -1 and prot_context_end != -1
    prot_oldchar_start = int(first_prot.prot_oldchar_start)
    prot_oldchar_end = int(first_prot.prot_oldchar_end)

    if prot_context_start == prot_oldchar_start and prot_context_end == prot_oldchar_end:
        trig_context_start = trig_oldchar_start
        trig_context_end = trig_oldchar_start+len(trig_name)

    elif trig_oldchar_start > prot_oldchar_end:
        relative_distance = trig_oldchar_start - prot_oldchar_end
        trig_context_start = prot_context_end + relative_distance
        trig_context_end = trig_context_start+len(trig_name)
        #print('---', context[trig_context_start:trig_context_end], '---', trig_name, '---')

    elif trig_oldchar_start == prot_oldchar_end:
        if old_sent[trig_oldchar_start-1:trig_oldchar_start] != " ":
            trig_context_start = prot_context_end
            trig_context_end = trig_context_start + len(trig_name)
        else:
            trig_context_start = prot_context_end-1
            trig_context_end = trig_context_start+len(trig_name)

    elif trig_oldchar_start == prot_oldchar_start:
        trig_context_start = prot_context_start
        trig_context_end += trig_context_start+len(trig_name)+1

    elif trig_oldchar_start+len(trig_name) < prot_oldchar_start:
        relative_distance = prot_oldchar_start - trig_oldchar_start
        trig_context_start = prot_context_start - relative_distance
        trig_context_end += trig_context_start+len(trig_name)+1

    elif prot_oldchar_start < trig_oldchar_start and trig_oldchar_start+len(trig_name) <= prot_oldchar_end:
        relative_distance = trig_oldchar_start - prot_oldchar_start
        trig_context_start = prot_context_start + relative_distance
        trig_context_end = trig_context_start+len(trig_name)

    elif trig_oldchar_start < prot_oldchar_start:
        relative_distance = prot_oldchar_start - trig_oldchar_start
        trig_context_start = prot_context_start-relative_distance
        trig_context_end = trig_context_start+len(trig_name)

    else:
        print("there is a serious problem!")

    if context[trig_context_start:trig_context_end] != trig_name:
        trig_context_start += 1
        trig_context_end += 1

    if context[trig_context_start:trig_context_end] != trig_name:
        print(file_name)
        print(old_sent)
        print(trig_name)
        print(trig_context_start, '---', trig_context_end)
        print('---', context[trig_context_start:trig_context_end], '---', trig_name, '---')
        print("#################$$$$$$$")

    return trig_context_start, trig_context_end

def gen_tags():
    entity_tags = []
    for type in constant.SIMP_TYPE + constant.BIND_TYPE + constant.PMOD_TYPE + constant.REGU_TYPE:
        entity_tags.append('<S:' + type + '>')
        entity_tags.append('</S:' + type + '>')
        entity_tags.append('<O:' + type + '>')
        entity_tags.append('</O:' + type + '>')

    entity_tags.append('<O:Protein>')
    entity_tags.append('</O:Protein>')

    return entity_tags

'''
predict中生成当前句子的 context join protein特征
sent_prots是句子中标注的蛋白质集合
prot_seq是标注的蛋白质序列
context_jprots是前面句子中出现的join protein
'''
def get_context_jport_BIO(word_set, prot_dict, prev_jprot_set):
    sent_len = len(word_set[0].split())
    cjprot_seqs = ["Other"] * sent_len

    for prot in prot_dict.values():
        prot_name = prot.prot_name
        prot_start = prot.prot_start
        prot_end = prot.prot_end
        if prot_name in prev_jprot_set:
            if prot_start == prot_end:
                cjprot_seqs[prot_start] = "B-Protein"
            else:
                for i in range(prot_start, prot_end + 1):
                    if i == prot_start:
                        cjprot_seqs[i] = "B-Protein"
                    else:
                        cjprot_seqs[i] = "I-Protein"
    return cjprot_seqs


'''
 生成 previous sentence trigger feature
 context_trig可能不会大, 因为篇章不是太长
 该函数需要调试
'''
import re
def get_context_trig_BIO(word_set, context_triginfo_dict):
    sent = word_set[0]
    sent_len = len(sent.split())
    ctrig_seqs = ["Other"] * sent_len
    trig_info_dict = sorted(context_triginfo_dict.items(), key=lambda item: len(item[0].split()), reverse=True)
    for trig_info in trig_info_dict:
        maybe_trig_name, maybe_trig_type = trig_info[0], trig_info[1]
        trig_num = len(maybe_trig_name.split())
        begin = 0
        while sent.find(maybe_trig_name, begin) != -1:
            char_start = sent.find(maybe_trig_name, begin)
            word_start = len(sent[0:char_start].split())
            if word_start < sent_len:
                if ctrig_seqs[word_start] == "Other":
                    for i in range(word_start, word_start+trig_num):
                        if i == word_start:
                            ctrig_seqs[i] = "B-"+maybe_trig_type
                        else:
                            ctrig_seqs[i] = "I-"+maybe_trig_type
            begin = char_start + len("you")

    return ctrig_seqs
## ---------------------------------------------------------------------------------------------------------
## 2月18号新更改, 将插入实体的标签在预处理数据的时候完成,在神经网络阶段只需要调用即可
## trig与argument 两个实体之间的关系判断
def gen_new_sentence(opt, words, prev_sent, next_sent, trig_type, trig_position, argu_type, argu_position):
    trig_start = trig_position.index(0)
    trig_end = trig_start + trig_position.count(0) - 1
    argu_start = argu_position.index(0)
    argu_end = argu_start + argu_position.count(0) - 1

    trig_start_tag = '<S:'+trig_type+'>'
    trig_end_tag = '</S:'+trig_type+'>'
    argu_start_tag = '<O:' + argu_type + '>'
    argu_end_tag = '</O:' + argu_type + '>'
    token = words.split()

    argu_tag_posit = -1
    trig_tag_posit = -1
    if argu_start > trig_end: ## 先插入论元的标签,再插入trigger的标签
        token.insert(argu_end+1, argu_end_tag)
        token.insert(argu_start, argu_start_tag)
        token.insert(trig_end+1, trig_end_tag)
        token.insert(trig_start, trig_start_tag)
        ## trigger 和 argument的开始标签的位置
        trig_tag_posit = trig_start
        argu_tag_posit = argu_start+2

    elif trig_start > argu_end:
        token.insert(trig_end + 1, trig_end_tag)
        token.insert(trig_start, trig_start_tag)
        token.insert(argu_end + 1, argu_end_tag)
        token.insert(argu_start, argu_start_tag)
        ## trigger 和 argument的新位置
        trig_tag_posit = trig_start+2
        argu_tag_posit = argu_start

    elif trig_start == argu_start and trig_end == argu_end: # <S:type> <O:type> word </O:type> </S:type>
        token.insert(argu_end + 1, argu_end_tag)
        token.insert(argu_start, argu_start_tag)
        token.insert(trig_end + 3, trig_end_tag)
        token.insert(trig_start, trig_start_tag)
        trig_tag_posit = trig_start
        argu_tag_posit = argu_start +1

    else:
        print(trig_start, '---', trig_end)
        print(argu_start, '---', argu_end)
        print("这是个什么情况")

    # # 为当前句子添加前后文
    # prev_tokens = prev_sent.split()
    # next_tokens = next_sent.split()
    # prev_wordnum = 0  # 在当期句子的前面增加了几个词
    # if len(token) < opt['context_window']:
    #     word_num = (opt['context_window'] - len(token)) // 2
    #     if prev_sent != "" and next_sent != "":
    #         token = prev_tokens[-word_num:] + token + next_tokens[0:word_num]
    #         prev_wordnum = min([word_num, len(prev_tokens)])
    #
    #     elif prev_sent != "" and next_sent == "":  # 文章的最后一句话, 句子中的实体位置改变, 向后移(2*word_num)
    #         token = prev_tokens[-2 * word_num:] + token
    #         prev_wordnum = min([2 * word_num, len(prev_tokens)])
    #
    #     elif prev_sent == "" and next_sent != "":  # 文章的第一句话, 句子中的实体位置不需要改变
    #         token = token + next_tokens[0:word_num]
    #
    # trig_tag_posit = trig_tag_posit+prev_wordnum
    # argu_tag_posit = argu_tag_posit+prev_wordnum

    ## 为当前句子添加前后句子
    prev_tokens = prev_sent.split()
    next_tokens = next_sent.split()
    if prev_sent != "" and next_sent != "":
        token = prev_tokens + token + next_tokens
        trig_tag_posit = get_positions(trig_tag_posit + len(prev_tokens), trig_tag_posit + len(prev_tokens), len(token))
        argu_tag_posit = get_positions(argu_tag_posit + len(prev_tokens), argu_tag_posit + len(prev_tokens), len(token))

    elif prev_sent != "" and next_sent == "":  # 文章的最后一句话, 句子中的实体位置改变, 向后移(2*word_num)
        token = prev_tokens + token
        trig_tag_posit = get_positions(trig_tag_posit + len(prev_tokens), trig_tag_posit + len(prev_tokens), len(token))
        argu_tag_posit = get_positions(argu_tag_posit + len(prev_tokens), argu_tag_posit + len(prev_tokens), len(token))

    elif prev_sent == "" and next_sent != "":  # 文章的第一句话, 句子中的实体位置不需要改变
        token = token + next_tokens
        trig_tag_posit = get_positions(trig_tag_posit, trig_tag_posit, len(token))
        argu_tag_posit = get_positions(argu_tag_posit, argu_tag_posit, len(token))

    new_word = " ".join(token)
    return new_word, trig_tag_posit, argu_tag_posit

'''
binding模块中三个实体的前后插入标签
regu模块三个实体的前后插入标签
'''
def gen_bindregu_sentence(opt, words, prev_sent, next_sent, trig_type, trig_position, argu1_type,  argu1_position, argu2_type, argu2_position): # argu1_type 和 argu2_type
    ## 得到开始和介绍位置
    trig_start = trig_position.index(0)
    trig_end = trig_start + trig_position.count(0) - 1

    argu1_start = argu1_position.index(0)
    argu1_end = argu1_start + argu1_position.count(0) - 1

    argu2_start = argu2_position.index(0)
    argu2_end = argu2_start + argu2_position.count(0) - 1
    token = words.split()

    start_posit = {'S:'+trig_type:(trig_start, trig_end), '1O:'+argu1_type: (argu1_start, argu1_end), '2O:'+argu2_type: (argu2_start, argu2_end)}
    sorted_result = sorted(start_posit.items(), key=lambda x: x[1][0], reverse=True)
    dist = 4
    tag_dict = dict()
    for item in sorted_result:
        tag_list = list(item[0])
        start_tag = ""
        end_tag = ""
        label = "" # 用来说明是trigger, argu1 or argu2
        if tag_list[0] == 'S':
            start_tag = '<'+item[0]+'>'
            end_tag = '</'+item[0]+'>'
            label = 'trigger'

        elif tag_list[0] == '1':
            start_tag = '<O:Protein>'
            end_tag = '</O:Protein>'
            label = 'first'
        elif tag_list[0] == '2':
            start_tag = '<O:Protein>'
            end_tag = '</O:Protein>'
            label = 'second'

        token.insert(item[1][1]+1, end_tag) #插入结束符tag
        token.insert(item[1][0], start_tag)
        tag_posit = item[1][0] + dist
        dist -= 2
        tag_dict[label] = tag_posit

    # 为当前句子添加前后文
    # prev_tokens = prev_sent.split()
    # next_tokens = next_sent.split()
    # prev_wordnum = 0  # 在当期句子的前面增加了几个词
    # if len(token) < opt['context_window']:
    #     word_num = (opt['context_window'] - len(token)) // 2
    #     if prev_sent != "" and next_sent != "":
    #         token = prev_tokens[-word_num:] + token + next_tokens[0:word_num]
    #         prev_wordnum = min([word_num, len(prev_tokens)])
    #
    #     elif prev_sent != "" and next_sent == "":  # 文章的最后一句话, 句子中的实体位置改变, 向后移(2*word_num)
    #         token = prev_tokens[-2 * word_num:] + token
    #         prev_wordnum = min([2 * word_num, len(prev_tokens)])
    #
    #     elif prev_sent == "" and next_sent != "":  # 文章的第一句话, 句子中的实体位置不需要改变
    #         token = token + next_tokens[0:word_num]
    ## 添加前后句子
    prev_tokens = prev_sent.split()
    next_tokens = next_sent.split()
    flag = False
    if prev_sent != "" and next_sent != "":
        token = prev_tokens + token + next_tokens
        flag = True
    elif prev_sent != "" and next_sent == "":  # 文章的最后一句话, 句子中的实体位置改变, 向后移(2*word_num)
        token = prev_tokens + token
        flag = True
    elif prev_sent == "" and next_sent != "":  # 文章的第一句话, 句子中的实体位置不需要改变
        token = token + next_tokens

    trig_tag_posit = -1
    argu1_tag_posit = -1
    argu2_tag_posit = -1
    for label, tag_posit in tag_dict.items():
        if flag == True:
            if label == 'trigger':
                trig_tag_posit = get_positions(tag_posit+len(prev_tokens), tag_posit+len(prev_tokens), len(token))
            elif label == 'first':
                argu1_tag_posit = get_positions(tag_posit+len(prev_tokens), tag_posit+len(prev_tokens), len(token))
            elif label == 'second':
                argu2_tag_posit = get_positions(tag_posit+len(prev_tokens), tag_posit+len(prev_tokens), len(token))
        elif flag == False:
            if label == 'trigger':
                trig_tag_posit = get_positions(tag_posit, tag_posit, len(token))
            elif label == 'first':
                argu1_tag_posit = get_positions(tag_posit, tag_posit, len(token))
            elif label == 'second':
                argu2_tag_posit = get_positions(tag_posit, tag_posit, len(token))

    new_word = " ".join(token)
    assert len(trig_tag_posit) == len(argu1_tag_posit)
    assert len(argu1_tag_posit) == len(argu2_tag_posit)

    return new_word, trig_tag_posit, argu1_tag_posit, argu2_tag_posit

## 仅仅判断两个论元的情况, 不插入trigger
def gen_bindregu_sentence_two(opt, words, prev_sent, next_sent, argu1_type, argu1_position, argu2_type, argu2_position): # argu1_type 和 argu2_type
    argu1_start = argu1_position.index(0)
    argu1_end = argu1_start + argu1_position.count(0) - 1

    argu2_start = argu2_position.index(0)
    argu2_end = argu2_start + argu2_position.count(0) - 1
    token = words.split()
    argu1_start_tag = "<"+argu1_type+">"
    argu1_end_tag = "</"+argu1_type+">"
    argu2_start_tag = "<" + argu2_type + ">"
    argu2_end_tag = "</" + argu2_type + ">"
    token.insert(argu2_end + 1, argu2_end_tag)
    token.insert(argu2_start, argu2_start_tag)
    token.insert(argu1_end + 1, argu1_end_tag)
    token.insert(argu1_start, argu1_start_tag)

    argu1_startag_posit = argu1_start
    argu2_startag_posit = argu2_start + 2

    # 为当前句子添加前后文
    # prev_tokens = prev_sent.split()
    # next_tokens = next_sent.split()
    # prev_wordnum = 0  # 在当期句子的前面增加了几个词
    # if len(token) < opt['context_window']:
    #     word_num = (opt['context_window'] - len(token)) // 2
    #     if prev_sent != "" and next_sent != "":
    #         token = prev_tokens[-word_num:] + token + next_tokens[0:word_num]
    #         prev_wordnum = min([word_num, len(prev_tokens)])
    #
    #     elif prev_sent != "" and next_sent == "":  # 文章的最后一句话, 句子中的实体位置改变, 向后移(2*word_num)
    #         token = prev_tokens[-2 * word_num:] + token
    #         prev_wordnum = min([2 * word_num, len(prev_tokens)])
    #
    #     elif prev_sent == "" and next_sent != "":  # 文章的第一句话, 句子中的实体位置不需要改变
    #         token = token + next_tokens[0:word_num]
    ## 添加前后句子
    prev_tokens = prev_sent.split()
    next_tokens = next_sent.split()
    if prev_sent != "" and next_sent != "":
        token = prev_tokens + token + next_tokens
        argu1_startag_posit = get_positions(argu1_startag_posit + len(prev_tokens), argu1_startag_posit + len(prev_tokens), len(token))
        argu2_startag_posit = get_positions(argu2_startag_posit + len(prev_tokens), argu2_startag_posit + len(prev_tokens), len(token))

    elif prev_sent != "" and next_sent == "":  # 文章的最后一句话, 句子中的实体位置改变, 向后移(2*word_num)
        token = prev_tokens + token
        argu1_startag_posit = get_positions(argu1_startag_posit + len(prev_tokens), argu1_startag_posit + len(prev_tokens), len(token))
        argu2_startag_posit = get_positions(argu2_startag_posit + len(prev_tokens), argu2_startag_posit + len(prev_tokens), len(token))

    elif prev_sent == "" and next_sent != "":  # 文章的第一句话, 句子中的实体位置不需要改变
        token = token + next_tokens
        argu1_startag_posit = get_positions(argu1_startag_posit, argu1_startag_posit, len(token))
        argu2_startag_posit = get_positions(argu2_startag_posit, argu2_startag_posit, len(token))

    new_word = " ".join(token)
    assert len(argu1_startag_posit) == len(argu2_startag_posit)

    return new_word, argu1_startag_posit, argu2_startag_posit
#####################################################################
'''
 在test语料中, 识别出的trigger和生成的event, 需要记录其的标识号
'''
def is_read_a1_txt(prev_file_name, file_name, curr_trig_idx, curr_context):
    trig_idx = 1
    context = ""
    if prev_file_name != file_name: ## 开始了一个新文件，要计算trig的idx
        dot_position = file_name.rfind(".")
        pure_name = file_name[0:dot_position]
        #with open("/home/fang/Downloads/bioNlp2011/2011_test_data" + "/" + pure_name + ".txt", 'r') as txt_reader:
        with open("/home/fang/myworks/corpus_process/corpus_items/dataset2013/origin_data/2013_test"+ "/" + pure_name + ".txt", 'r') as txt_reader:
            context = txt_reader.read()
        #with open("/home/fang/Downloads/bioNlp2011/2011_test_data"+"/"+pure_name+".a1", 'r') as a1_reader:
        with open("/home/fang/myworks/corpus_process/corpus_items/dataset2013/origin_data/2013_test"+"/"+pure_name+".a1", 'r') as a1_reader:
            lines = a1_reader.readlines()
            prot_num = len(lines)
            if prot_num == 0:
                return 0, context

            prot_digit = lines[-1].split("\t")[0][1:]  #T17	Protein 1757 1762	IRF-4
            # print(file_name)
            # print(prot_num, "-----", int(prot_digit))
            # print("########################################")
            #assert prot_num == int(prot_digit)
            trig_idx = int(prot_digit)+1

    elif prev_file_name == file_name:
        trig_idx = curr_trig_idx
        context = curr_context
    else:
        print("no other situation about trigger")
    return trig_idx, context

'''
## 计算event_idx
file_name是list
'''
def record_event_idx(prev_file_name, file_name, curr_event_idx):
    event_idx = 1
    if prev_file_name == "" or prev_file_name != file_name: ## 刚开始或者开启一个新文件
        event_idx = 1
    elif file_name == prev_file_name:
        event_idx = curr_event_idx
    else:
        print("no other situation about event")
    return event_idx
