from utils import constant, common_utils
from data_prepare.entity import Protein, Trigger, Event
import itertools
from data.bind_example import gen_bind_example
from data.regu_example import gen_regu_example

'''
针对train
一句话生成的训练样例,正样例和负样例
把嵌套事件简化为Trgger与Trigger之间的关系
'''
def gen_sent_pairs(opt, words, prev_sent, next_sent, trig_dict, prot_dict, event_dict):
    sent_len = len(words.split())
    entity_pairs_first = list()
    entity_pairs_second = list() ## 这里存储的是关系判断的输入, new_words, new_trig_posit, new_argu_posit
    entity_exams = list()  ## 保留old_word, old_trig_posit, old_argu_posit
    pair_labels = list()

    simp_trigs = list()
    bind_trigs = list()
    pmod_trigs = list()
    regu_trigs = list()

    #将trigger词分为三类
    for trig in trig_dict.values():
        trig_type = trig.trig_type
        if trig_type in constant.SIMP_TYPE:
            simp_trigs.append(trig)
        elif trig_type in constant.BIND_TYPE:
            bind_trigs.append(trig)
        elif trig_type in constant.PMOD_TYPE:
            pmod_trigs.append(trig)
        elif trig_type in constant.REGU_TYPE:
            regu_trigs.append(trig)
        else:
            print('data_utils.py error')

    # 是不是应该以trigger出现的顺序生成pair, trigger的出现顺序为前提
    # 所有trig与prot逐一配对
    for trig in simp_trigs + bind_trigs + pmod_trigs + regu_trigs:
        trig_start = trig.trig_start
        trig_end = trig.trig_end
        trig_posit = get_positions(trig_start, trig_end, sent_len)
        trig_type_idx = [constant.NER_TO_ID[trig.trig_type]]
        for prot in prot_dict.values():
            argu_posit = get_positions(prot.prot_start, prot.prot_end, sent_len)
            argu_type_idx = [constant.NER_TO_ID['Protein']]

            new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                      words,
                                                                                      prev_sent,
                                                                                      next_sent,
                                                                                      trig.trig_type,
                                                                                      trig_posit,
                                                                                      'Protein',
                                                                                      argu_posit)

            event_idx_set, rela_type = get_relatype(trig, prot, event_dict)
            entity_pairs_first.append((trig_type_idx, trig_posit, argu_type_idx, argu_posit))
            entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit)) #rela_typeId是trigger与prot的关系
            entity_exams.append((event_idx_set, words, trig, prot, trig_posit, argu_posit))
            pair_labels.append(rela_type)

    # comp_trig与simp_trig和bind_trig组成的触发词对
    for trig in regu_trigs:
        trig_start = trig.trig_start
        trig_end = trig.trig_end
        trig_posit = get_positions(trig_start, trig_end, sent_len)
        trig_type_idx = [constant.NER_TO_ID[trig.trig_type]]
        for simp_t in simp_trigs:
            argu_start = simp_t.trig_start
            argu_end = simp_t.trig_end
            argu_posit = get_positions(argu_start, argu_end, sent_len)
            argu_type_idx = [constant.NER_TO_ID[simp_t.trig_type]]

            if argu_start > trig_end or trig_start > argu_end:
                new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                          words,
                                                                                          prev_sent,
                                                                                          next_sent,
                                                                                          trig.trig_type,
                                                                                          trig_posit,
                                                                                          simp_t.trig_type,
                                                                                          argu_posit)
                event_idx_set, rela_type = get_relatype(trig, simp_t, event_dict)
                entity_pairs_first.append(((trig_type_idx, trig_posit, argu_type_idx, argu_posit)))
                entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit))
                entity_exams.append((event_idx_set, words, trig, simp_t, trig_posit, argu_posit))
                pair_labels.append(rela_type)

        for bind_t in bind_trigs:
            argu_start = bind_t.trig_start
            argu_end = bind_t.trig_end
            argu_posit = get_positions(argu_start, argu_end, sent_len)
            argu_type_idx = [constant.NER_TO_ID[bind_t.trig_type]]
            if argu_start > trig_end or trig_start > argu_end:
                new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                          words,
                                                                                          prev_sent,
                                                                                          next_sent,
                                                                                          trig.trig_type,
                                                                                          trig_posit,
                                                                                          bind_t.trig_type,
                                                                                          argu_posit)

                event_idx_set, rela_type = get_relatype(trig, bind_t, event_dict)
                entity_pairs_first.append(((trig_type_idx, trig_posit, argu_type_idx, argu_posit)))
                entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit))
                entity_exams.append((event_idx_set, words, trig, bind_t, trig_posit, argu_posit))
                pair_labels.append(rela_type)

        for pmod_t in pmod_trigs:
            argu_start = pmod_t.trig_start
            argu_end = pmod_t.trig_end
            argu_posit = get_positions(argu_start, argu_end, sent_len)
            argu_type_idx = [constant.NER_TO_ID[pmod_t.trig_type]]
            if argu_start > trig_end or trig_start > argu_end:
                new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                          words,
                                                                                          prev_sent,
                                                                                          next_sent,
                                                                                          trig.trig_type,
                                                                                          trig_posit,
                                                                                          pmod_t.trig_type,
                                                                                          argu_posit)

                event_idx_set, rela_type = get_relatype(trig, pmod_t, event_dict)
                entity_pairs_first.append(((trig_type_idx, trig_posit, argu_type_idx, argu_posit)))
                entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit))
                entity_exams.append((event_idx_set, words, trig, pmod_t, trig_posit, argu_posit))
                pair_labels.append(rela_type)

    #既要生成AB，又要生成BA(此处注意, 一定选择用permutations)
    regu_pairs = list(itertools.permutations(regu_trigs, 2))
    for r_pair in regu_pairs:
        trig, argu_trig = r_pair
        trig_start = trig.trig_start
        trig_end = trig.trig_end
        trig_posit = get_positions(trig_start, trig_end, sent_len)
        trig_type_idx = [constant.NER_TO_ID[trig.trig_type]]

        argu_start = argu_trig.trig_start
        argu_end = argu_trig.trig_end
        argu_posit = get_positions(argu_start, argu_end, sent_len)
        argu_type_idx = [constant.NER_TO_ID[argu_trig.trig_type]]

        if trig_start > argu_end or argu_start > trig_end:
            new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                      words,
                                                                                      prev_sent,
                                                                                      next_sent,
                                                                                      trig.trig_type,
                                                                                      trig_posit,
                                                                                      argu_trig.trig_type,
                                                                                      argu_posit)

            event_idx_set, rela_type = get_relatype(trig, argu_trig, event_dict)
            entity_pairs_first.append(((trig_type_idx, trig_posit, argu_type_idx, argu_posit)))
            entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit))
            entity_exams.append((event_idx_set, words, trig, argu_trig, trig_posit, argu_posit))
            pair_labels.append(rela_type)

    entity_pair_info = (entity_pairs_first, entity_pairs_second, pair_labels, entity_exams)
    # 针对当前句子生成binding类型的训练语料
    bind_inputs_first, bind_inputs_second, bind_labels, bind_exams = gen_bind_example(opt, words, prev_sent, next_sent, entity_exams, pair_labels, "train")
    bind_argus = (bind_inputs_first, bind_inputs_second, bind_labels, bind_exams)
    # 针对regulation 三个事件类型生成训练语料,同 binding类型一起训练
    protmod_argus, regu_argus = gen_regu_example(opt, words, prev_sent, next_sent, entity_exams, pair_labels, "train")

    return entity_pair_info, bind_argus, protmod_argus, regu_argus

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + list(range(1, length-end_idx))

'''
有些实体对出现中多个事件中，所出现过的事件event_idx都记录下来
'''
def get_relatype(trig_entity, argu_entity, event_dict):
    trig_idx = trig_entity.trig_idx
    argu_idx = ''
    if isinstance(argu_entity, Trigger):
        argu_idx = argu_entity.trig_idx
    elif isinstance(argu_entity, Protein):
        argu_idx = argu_entity.prot_idx
    assert argu_idx != ''

    #判断 trig 与 argu 的关系
    event_idx_set = list()
    relatype_set = list()
    for event in event_dict.values():
        rela_type = has_rela(event, trig_idx, argu_idx, event_dict)
        if rela_type != "Other":
            event_idx = event.event_idx
            event_idx_set.append(event_idx)
            relatype_set.append(rela_type)
    if len(event_idx_set) > 0:
        assert len(list(set(relatype_set))) == 1
        return event_idx_set, relatype_set[0]
    else:
        assert len(event_idx_set) == 0
        return event_idx_set, "Other"

'''
 判断trig与argu是否在一个event中
 如果 trig与 argu在同一个event中, 将event的 idx 提取出来(可能存在与多个event中)
 '''
def has_rela(event, trig_idx, argu_idx, event_dict):
    event_trig_idx = event.event_trig_idx
    fargu_idx = event.first_argu_idx
    sargu_idx = event.second_argu_idx
    if sargu_idx == "":
        if fargu_idx.startswith("T"):
            if trig_idx == event_trig_idx and argu_idx == fargu_idx:
                return event.first_argu_type
        elif fargu_idx.startswith("E"):
            argutrig_idx = get_argutrig_idx(fargu_idx, event_dict)
            if trig_idx == event_trig_idx and argu_idx == argutrig_idx:
                return event.first_argu_type

    elif sargu_idx != "":
        if fargu_idx.startswith("T"):
            if trig_idx == event_trig_idx and argu_idx == fargu_idx:
                return event.first_argu_type
        elif fargu_idx.startswith("E"):
            fargutrig_idx = get_argutrig_idx(fargu_idx, event_dict)
            if trig_idx == event_trig_idx and argu_idx == fargutrig_idx:
                return event.first_argu_type
        if sargu_idx.startswith("T"):
            if trig_idx == event_trig_idx and argu_idx == sargu_idx:
                assert event.second_argu_type in ['Theme', 'Cause']
                return event.second_argu_type
        elif sargu_idx.startswith("E"):
            sargutrig_idx = get_argutrig_idx(sargu_idx, event_dict)
            if trig_idx == event_trig_idx and argu_idx == sargutrig_idx:
                assert event.second_argu_type in ['Theme', 'Cause']
                return event.second_argu_type

    if len(event.other_argu_info) > 0:
        for other_argu_idx, other_argu_type in event.other_argu_info.items():
            if trig_idx == event_trig_idx and argu_idx == other_argu_idx:
                return other_argu_type

    return "Other"

## 根据event_idx在event_dict中查找对应的event
def get_argutrig_idx(event_id, event_dict):
    argu_event = event_dict[event_id]
    arguevent_trigId = argu_event.event_trig_idx
    assert arguevent_trigId != ''
    return arguevent_trigId

## 统计regu事件Cause论元的个数（一个句子中）
def sent_cause_num(event_dict):
    protargu_cause = list()
    evetargu_cause = list()

    for evet_idx, evet in event_dict.items():
        trigger_idx = evet.event_trig_idx
        second_argu = evet.second_argu
        second_argu_type = evet.second_argu_type
        if second_argu != None and second_argu_type == 'Cause':
            if isinstance(second_argu, Protein):
                prot_idx = second_argu.prot_idx
                protargu_cause.append(trigger_idx+"_"+prot_idx)

            elif isinstance(second_argu, Event):
                argu_idx = second_argu.event_trig_idx
                evetargu_cause.append(trigger_idx+"_"+argu_idx)
    return len(set(protargu_cause)), len(set(evetargu_cause))

def get_prot_name(words, pred_jprot_seq):
    pred_protname_set = []
    tokens = words[0].split()
    assert len(tokens) == len(pred_jprot_seq)
    entity_list = get_ner_BIO(pred_jprot_seq)
    for label in entity_list:
        market = label.index(']')
        if ',' in label[1:market]:
            trig_start = int(label[1:market].split(",")[0])
            trig_end = int(label[1:market].split(",")[1])
        else:
            trig_start = int(label[1:market])
            trig_end = int(label[1:market])
        prot_name = " ".join(tokens[trig_start:trig_end + 1])
        pred_protname_set.append(prot_name)
    return pred_protname_set

## 使用了BIO标签系统
def gen_trig_dict_BIO(words, trig_seq, prot_num):
    trig_dict = dict()
    predtrig_nametype_dict = dict()
    trig_idx = prot_num+1
    tokens = words[0].split()
    assert len(tokens) == len(trig_seq)
    entity_list = get_ner_BIO(trig_seq)
    for label in entity_list:
        market = label.index(']')
        trig_type = label[market+1:].lower().capitalize()
        if ',' in label[1:market]:
            trig_start = int(label[1:market].split(",")[0])
            trig_end = int(label[1:market].split(",")[1])
        else:
            trig_start = int(label[1:market])
            trig_end = int(label[1:market])
        trig_name = " ".join(tokens[trig_start:trig_end+1])
        predtrig_nametype_dict[trig_name] = trig_type
        trig_entity = Trigger("T" + str(trig_idx), trig_name, trig_type, None, None, None, None, trig_start, trig_end)
        trig_dict["T" + str(trig_idx)] = trig_entity
        trig_idx += 1
    return trig_dict, predtrig_nametype_dict

'''
  与 gen_trig_dict_bio的区别是最后一个传入的参数
  用在train.py 中的test函数中
'''
def gen_testtrig_dict_bio(words, trig_labels, trig_idx):
    trig_dict = dict()
    predtrig_nametype_dict = dict()
    tokens = words[0].split()
    assert len(tokens) == len(trig_labels)
    entity_list = get_ner_BIO(trig_labels)
    for label in entity_list:
        market = label.index(']')
        trig_type = label[market + 1:].lower().capitalize()
        if ',' in label[1:market]:
            trig_start = int(label[1:market].split(",")[0])
            trig_end = int(label[1:market].split(",")[1])
        else:
            trig_start = int(label[1:market])
            trig_end = int(label[1:market])
        trig_name = " ".join(tokens[trig_start:trig_end + 1])
        predtrig_nametype_dict[trig_name] = trig_type
        trig_entity = Trigger("T" + str(trig_idx), trig_name, trig_type, None, None, None, None, trig_start, trig_end)
        trig_dict["T" + str(trig_idx)] = trig_entity

        trig_idx += 1
    return trig_dict, predtrig_nametype_dict, trig_idx

## 识别join protein使用了BIO标签系统
## prot_dict是标注的蛋白质的位置
def gen_jprot_dict_BIO(jprot_seq, prot_dict):
    jprot_dict = dict()
    prot_set = get_ner_BIO(jprot_seq)
    for prot in prot_set:
        locate = prot.find("]")
        assert locate != -1
        if ',' in prot[1:locate]:
            prot_start = int(prot[1:locate].split(",")[0])
            prot_end = int(prot[1:locate].split(",")[1])
        else:
            prot_start = int(prot[1:locate])
            prot_end = int(prot[1:locate])

        for prot_idx, prot in prot_dict.items():
            origin_prot_start = prot.prot_start
            origin_prot_end = prot.prot_end
            if origin_prot_start == prot_start and origin_prot_end == prot_end:
                jprot_dict[prot_idx] = prot
                break
    return jprot_dict

## 从BIO标签系统中抽取实体
def get_ner_BIO(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string
'''
development 和 test corpus
train:(event_idx_set, trig, trig_type_idx, trig_posit, argu_trig, argu_type_idx, argu_posit, rela_type)
devel:(None, trig, trig_type_idx, trig_posit, prot, argu_type_idx, argu_posit)
'''
def gen_entity_pairs(opt, words, prev_sent, next_sent, trig_dict, prot_dict):
    entity_pairs_first = list()
    entity_pairs_second = list()
    entity_pairs_exams = list()

    simp_trigs = list()
    bind_trigs = list()
    pmod_trigs = list()
    regu_trigs = list()

    #将trigger词分为三类
    for trig in trig_dict.values():
        trig_type = trig.trig_type
        if trig_type in constant.SIMP_TYPE:
            simp_trigs.append(trig)
        elif trig_type in constant.BIND_TYPE:
            bind_trigs.append(trig)
        elif trig_type in constant.PMOD_TYPE:
            pmod_trigs.append(trig)
        elif trig_type in constant.REGU_TYPE:
            regu_trigs.append(trig)
        else:
            print('data_utils.py error, 230行出错, ---', trig_type)

    # 是不是应该以trigger出现的顺序生成pair, 待考虑
    # 所有trig与prot逐一配对
    sent_len = len(words.split())
    for trig in simp_trigs + bind_trigs + pmod_trigs + regu_trigs:
        trig_start = trig.trig_start
        trig_end = trig.trig_end
        trig_posit = get_positions(trig.trig_start, trig.trig_end, sent_len)
        trig_type_idx = constant.NER_TO_ID[trig.trig_type]
        for prot in prot_dict.values():
            argu_start = prot.prot_start
            argu_end = prot.prot_end
            argu_type_idx = constant.NER_TO_ID['Protein']
            if trig_start > argu_end or argu_start > trig_end:
                argu_posit = get_positions(prot.prot_start, prot.prot_end, sent_len)
                new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                          words,
                                                                                          prev_sent,
                                                                                          next_sent,
                                                                                          trig.trig_type,
                                                                                          trig_posit,
                                                                                          "Protein",
                                                                                          argu_posit)

                entity_pairs_first.append(([trig_type_idx], trig_posit, [argu_type_idx], argu_posit))
                entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit))
                entity_pairs_exams.append((None, words, trig, prot, trig_posit, argu_posit))

    # comp_trig与simp_trig和bind_trig组成的触发词对
    for trig in regu_trigs:
        trig_start = trig.trig_start
        trig_end = trig.trig_end
        trig_posit = get_positions(trig.trig_start, trig.trig_end, sent_len)
        trig_type_idx = constant.NER_TO_ID[trig.trig_type]
        for simp_t in simp_trigs:
            argu_start = simp_t.trig_start
            argu_end = simp_t.trig_end
            argu_type_idx = constant.NER_TO_ID[simp_t.trig_type]
            if trig_start > argu_end or argu_start > trig_end:
                argu_posit = get_positions(simp_t.trig_start, simp_t.trig_end, sent_len)
                new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                          words,
                                                                                          prev_sent,
                                                                                          next_sent,
                                                                                          trig.trig_type,
                                                                                          trig_posit,
                                                                                          simp_t.trig_type,
                                                                                          argu_posit)
                entity_pairs_first.append(([trig_type_idx], trig_posit, [argu_type_idx], argu_posit))
                entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit))
                entity_pairs_exams.append((None, words, trig, simp_t, trig_posit, argu_posit))

        for bind_t in bind_trigs:
            argu_start = bind_t.trig_start
            argu_end = bind_t.trig_end
            argu_type_idx = constant.NER_TO_ID[bind_t.trig_type]
            if trig_start > argu_end or argu_start > trig_end:
                argu_posit = get_positions(bind_t.trig_start, bind_t.trig_end, sent_len)
                new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                          words,
                                                                                          prev_sent,
                                                                                          next_sent,
                                                                                          trig.trig_type,
                                                                                          trig_posit,
                                                                                          bind_t.trig_type,
                                                                                          argu_posit)
                entity_pairs_first.append(([trig_type_idx], trig_posit, [argu_type_idx], argu_posit))
                entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit))
                entity_pairs_exams.append((None, words, trig, bind_t, trig_posit, argu_posit))

        for pmod_t in pmod_trigs:
            argu_start = pmod_t.trig_start
            argu_end = pmod_t.trig_end
            argu_type_idx = constant.NER_TO_ID[pmod_t.trig_type]
            if trig_start > argu_end or argu_start > trig_end:
                argu_posit = get_positions(pmod_t.trig_start, pmod_t.trig_end, sent_len)
                new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                          words,
                                                                                          prev_sent,
                                                                                          next_sent,
                                                                                          trig.trig_type,
                                                                                          trig_posit,
                                                                                          pmod_t.trig_type,
                                                                                          argu_posit)

                entity_pairs_first.append(([trig_type_idx], trig_posit, [argu_type_idx], argu_posit))
                entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit))
                entity_pairs_exams.append((None, words, trig, pmod_t, trig_posit, argu_posit))


    #既要生成AB，又要生成BA
    regu_pairs = list(itertools.permutations(regu_trigs, 2))
    for r_pair in regu_pairs:
        trig, argu_trig = r_pair
        trig_start = trig.trig_start
        trig_end = trig.trig_end
        argu_start = argu_trig.trig_start
        argu_end = argu_trig.trig_end
        trig_posit = get_positions(trig.trig_start, trig.trig_end, sent_len)
        trig_type_idx = constant.NER_TO_ID[trig.trig_type]
        if trig_start > argu_end or argu_start > trig_end:
            argu_posit = get_positions(argu_trig.trig_start, argu_trig.trig_end, sent_len)
            argu_type_idx = constant.NER_TO_ID[argu_trig.trig_type]
            new_words, new_trig_posit, new_argu_posit = common_utils.gen_new_sentence(opt,
                                                                                      words,
                                                                                      prev_sent,
                                                                                      next_sent,
                                                                                      trig.trig_type,
                                                                                      trig_posit,
                                                                                      argu_trig.trig_type,
                                                                                      argu_posit)
            entity_pairs_first.append(([trig_type_idx], trig_posit, [argu_type_idx], argu_posit))
            entity_pairs_second.append((new_words, new_trig_posit, new_argu_posit))
            entity_pairs_exams.append((None, words, trig, argu_trig, trig_posit, argu_posit))

    return entity_pairs_first, entity_pairs_second, entity_pairs_exams

# 计算的结果需要赋值给utils.constant中的TRIGGER_TO_ID, 赋值操作在event_parsing.py中
def gen_trigidx():
    trigtype_dict = dict()
    trigtype_dict['Other'] = 0
    #trig_type = ['Gene_expression', 'Transcription', 'Phosphorylation', 'Protein_catabolism', 'Localization', 'Binding', 'Regulation', 'Positive_regulation', 'Negative_regulation']
    trig_type = constant.SIMP_TYPE + constant.BIND_TYPE + constant.PMOD_TYPE + constant.REGU_TYPE
    i = 1
    for prefix in ['B-', 'I-']:
        for type in trig_type:
            trigtype_dict[prefix+type] = i
            i += 1
    trigtype_dict['START'] = i
    trigtype_dict['STOP'] = i+1
    return trigtype_dict