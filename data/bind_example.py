from utils import constant, common_utils
import itertools
from metric.metric_util import trig_isequal
from data_prepare.entity import Protein

## devel 和 test 中对 binding event 生成语料
## entity_pairs: event_idx_set, trig, trig_position, argu, argu_position, tp_rela_id
## train语料
def gen_bind_example(opt, old_words, prev_sent, next_sent, entity_pairs, pair_labels, mode):
    #assert len(entity_pairs) > 0
    bind_example = dict()
    for pair, pred_label in zip(entity_pairs, pair_labels):
        event_idx_set, words, trig, argu, trig_posit, argu_posit = pair
        new_pair = (event_idx_set, trig, trig_posit, argu, argu_posit, pred_label) ## label是trig 与prot是否有关系预测的标签
        trig_type = trig.trig_type
        trig_idx = trig.trig_idx
        assert isinstance(pred_label, str)
        if pred_label == 'Theme' and trig_type in constant.BIND_TYPE:
            store_info(trig_idx, new_pair, bind_example)

    ## 生成 trig_type, trig_posit, argu1_type, prot1_posit, argu2_type, prot2_posit = exam
    bind_pair_inputs_first = list()
    bind_pair_inputs_second = list() # 这个是训练第三个模型的输入
    bind_pair_labels = list()  # 训练第三个模块的gold_label
    bind_pair_exams = list()
    for trig_idx, bind_exam_set in bind_example.items():
        if len(bind_exam_set) > 1:
            bind_pair_set = list(itertools.combinations(bind_exam_set, 2))
            for (pair1, pair2) in bind_pair_set:
                event_idx_set1, trig1, trig1_position, argu1, argu1_position, label1 = pair1
                event_idx_set2, trig2, trig2_position, argu2, argu2_position, label2 = pair2
                trig_type_str = trig1.trig_type
                assert trig_isequal(trig1, trig2) and trig_type_str == "Binding"
                assert isinstance(argu1, Protein) and isinstance(argu2, Protein)

                argu1_type_id = constant.NER_TO_ID["Protein"]
                argu2_type_id = constant.NER_TO_ID["Protein"]
                # Binding 的两个论元按在句子中的顺序出现
                if argu1.prot_start < argu2.prot_start:
                    new_words, argu1_mask, argu2_mask = common_utils.gen_bindregu_sentence_two(opt,
                                                                                               old_words,
                                                                                               prev_sent,
                                                                                               next_sent,
                                                                                               "Protein",
                                                                                               argu1_position,
                                                                                               "Protein",
                                                                                               argu2_position)

                    bind_pair_inputs_first.append(([argu1_type_id], argu1_position, [argu2_type_id], argu2_position))
                    bind_pair_inputs_second.append((new_words, argu1_mask, argu2_mask))
                    bind_pair_exams.append((trig1, argu1, label1, argu2, label2))

                else:
                    new_words, argu2_mask, argu1_mask = common_utils.gen_bindregu_sentence_two(opt,
                                                                                               old_words,
                                                                                               prev_sent,
                                                                                               next_sent,
                                                                                               "Protein",
                                                                                               argu2_position,
                                                                                               "Protein",
                                                                                               argu1_position)

                    bind_pair_inputs_first.append(([argu2_type_id], argu2_position, [argu1_type_id], argu1_position))
                    bind_pair_inputs_second.append((new_words, argu2_mask, argu1_mask))
                    bind_pair_exams.append((trig1, argu2, label2, argu1, label1))

                if mode == "train":
                    mark = is_event(event_idx_set1, event_idx_set2)  ##判断两个论元实体是否出现在一个事件中
                    bind_pair_labels.append(mark)

    if mode == "train":
        return bind_pair_inputs_first, bind_pair_inputs_second, bind_pair_labels, bind_pair_exams
    else:
        return bind_pair_inputs_first, bind_pair_inputs_second, bind_pair_exams

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