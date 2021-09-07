from utils import constant, common_utils
from metric.metric_util import trig_isequal
from data_prepare.entity import Protein, Trigger

def gen_regu_example(opt, old_words, prev_sent, next_sent, entity_pairs, pair_labels, mode):
    #assert len(entity_pairs) > 0
    regu_theme_example = dict()
    regu_cause_example = dict()

    pmod_theme_example = dict()
    pmod_cause_example = dict()

    for pair, pred_label in zip(entity_pairs, pair_labels):
        event_idx_set, words, trig, argu, trig_posit, argu_posit = pair
        new_pair = (event_idx_set, trig, trig_posit, argu, argu_posit, pred_label) ## label是trig 与prot是否有关系预测的标签
        trig_type = trig.trig_type
        trig_idx = trig.trig_idx
        assert isinstance(pred_label, str)
        if trig_type in constant.REGU_TYPE and pred_label == 'Theme':
            store_info(trig_idx, new_pair, regu_theme_example)
        elif trig_type in constant.REGU_TYPE and pred_label == 'Cause':
            store_info(trig_idx, new_pair, regu_cause_example)
        elif trig_type in constant.PMOD_TYPE and pred_label == "Theme":
            store_info(trig_idx, new_pair, pmod_theme_example)
        elif trig_type in constant.PMOD_TYPE and pred_label == "Cause":
            store_info(trig_idx, new_pair, pmod_cause_example)

    if mode == "train":
        pmod_pairs_first, pmod_pairs_second, pmod_pairs_labels, pmod_pairs_exams = relation_extract(opt, old_words, prev_sent, next_sent, pmod_theme_example, pmod_cause_example, mode)
        regu_pairs_first, regu_pairs_second, regu_pairs_labels, regu_pairs_exams = relation_extract(opt, old_words, prev_sent, next_sent, regu_theme_example, regu_cause_example, mode)

        return (pmod_pairs_first, pmod_pairs_second, pmod_pairs_labels, pmod_pairs_exams), (regu_pairs_first, regu_pairs_second, regu_pairs_labels, regu_pairs_exams)
    else:
        pmod_pairs_first, pmod_pairs_second, pmod_pairs_exams = relation_extract(opt, old_words, prev_sent, next_sent, pmod_theme_example, pmod_cause_example, mode)
        regu_pairs_first, regu_pairs_second, regu_pairs_exams = relation_extract(opt, old_words, prev_sent, next_sent, regu_theme_example, regu_cause_example, mode)

        return (pmod_pairs_first, pmod_pairs_second, pmod_pairs_exams), (regu_pairs_first, regu_pairs_second, regu_pairs_exams)

def relation_extract(opt, old_words, prev_sent, next_sent, theme_example, cause_example, mode):
    pair_inputs_first = list()
    pair_inputs_second = list()  # 这个是训练第三个模型的输入
    pair_labels = list()  # 训练第三个模块的gold_label
    pair_exams = list()

    for trig_idx, theme_exam_set in theme_example.items():
        if len(theme_exam_set) >= 1 and len(cause_example) >= 1:
            for regu_theme_pair in theme_exam_set:
                event_idx_set1, trig1, trig1_position, argu1, argu1_position, label1 = regu_theme_pair
                argu1_start = -1
                if isinstance(argu1, Protein):
                    argu1_type = "Protein"
                    argu1_type_id = [constant.NER_TO_ID[argu1_type]]
                    argu1_start = argu1.prot_start
                elif isinstance(argu1, Trigger):
                    argu1_type = argu1.trig_type
                    argu1_type_id = [constant.NER_TO_ID[argu1_type]]
                    argu1_start = argu1.trig_start

                if trig_idx in cause_example.keys():
                    regu_cause_exam_set = cause_example[trig_idx]
                    for regu_cause_pair in regu_cause_exam_set:
                        event_idx_set2, trig2, trig2_position, argu2, argu2_position, label2 = regu_cause_pair

                        trig_type_str = trig1.trig_type
                        assert trig_isequal(trig1, trig2) and trig_type_str in constant.REGU_TYPE + constant.PMOD_TYPE
                        argu2_start = -1
                        if isinstance(argu2, Protein):
                            argu2_type = "Protein"
                            argu2_type_id = [constant.NER_TO_ID[argu2_type]]
                            argu2_start = argu2.prot_start
                        elif isinstance(argu2, Trigger):
                            argu2_type = argu2.trig_type
                            argu2_type_id = [constant.NER_TO_ID[argu2_type]]
                            argu2_start = argu2.trig_start

                        if argu1_start < argu2_start:
                            new_words, argu1_mask, argu2_mask = common_utils.gen_bindregu_sentence_two(opt,
                                                                                                       old_words,
                                                                                                       prev_sent,
                                                                                                       next_sent,
                                                                                                       argu1_type,
                                                                                                       argu1_position,
                                                                                                       argu2_type,
                                                                                                       argu2_position)

                            pair_inputs_first.append((argu1_type_id, argu1_position, argu2_type_id, argu2_position))
                            pair_inputs_second.append((new_words, argu1_mask, argu2_mask))
                            pair_exams.append((trig1, argu1, label1, argu2, label2))
                        else:
                            new_words, argu2_mask, argu1_mask = common_utils.gen_bindregu_sentence_two(opt,
                                                                                                       old_words,
                                                                                                       prev_sent,
                                                                                                       next_sent,
                                                                                                       argu2_type,
                                                                                                       argu2_position,
                                                                                                       argu1_type,
                                                                                                       argu1_position)

                            pair_inputs_first.append((argu1_type_id, argu1_position, argu2_type_id, argu2_position))
                            pair_inputs_second.append((new_words, argu1_mask, argu2_mask))
                            pair_exams.append((trig1, argu1, label1, argu2, label2))

                        if mode == "train":
                            mark = is_event(event_idx_set1, event_idx_set2)  ##判断两个论元实体是否出现在一个事件中
                            pair_labels.append(mark)

    if mode == "train":
        return pair_inputs_first, pair_inputs_second, pair_labels, pair_exams
    else:
        return pair_inputs_first, pair_inputs_second, pair_exams


def store_info(trig_idx, pair, exam_dict):
    if trig_idx in exam_dict.keys():
        example = exam_dict[trig_idx]
        example.append(pair)
    else:
        exam_dict[trig_idx] = [pair]

## event_idx_set1和event_idx_set2如果有重合，说明可以组合成一个事件
def is_event(event_idx_set1, event_idx_set2):
    if len(event_idx_set1) > 0 and len(event_idx_set2) > 0:
        coincide = set([x for x in event_idx_set1 if x in event_idx_set2])
        if len(coincide) > 0:
            return 1
        else:
            return 0
    else:
        return 0