import metric.metric_util as util
from collections import Counter
import utils.constant as constant
import sys

#需要注意的是：gold_set是list，存储的元素是类：sent_event
# 存储gold_result和转换pred_result一定要注意格式
def score_event(pred_gold_events):
    corrt_events = Counter()
    gold_events = Counter()
    pred_events = Counter()

    pred_number_count = Counter()
    gold_number_count = Counter()

    for sent_pred_events, sent_gold_events in pred_gold_events:
        sent_event_eval(sent_pred_events, sent_gold_events, corrt_events, gold_events, pred_events, pred_number_count, gold_number_count)
    # 关于复杂事件标注和预测的个数:
    # print(pred_number_count)
    # print(gold_number_count)
    #结果评测
    # Print verbose information
    print("Per-relation statistics:")
    relations = gold_events.keys()
    longest_relation = 0
    for relation in sorted(relations):
        longest_relation = max(len(relation), longest_relation)
    for relation in sorted(relations):
        # (compute the score)
        corrt = corrt_events[relation]
        pred = pred_events[relation]
        gold = gold_events[relation]
        prec = 1.0
        if pred > 0:
            prec = float(corrt) / float(pred)
        recall = 0.0
        if gold > 0:
            recall = float(corrt) / float(gold)
        f1 = 0.0
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        # (print the score)
        sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
        sys.stdout.write("  P: ")
        if prec < 0.1: sys.stdout.write(' ')
        if prec < 1.0: sys.stdout.write(' ')
        sys.stdout.write("{:.2%}".format(prec))
        sys.stdout.write("  R: ")
        if recall < 0.1: sys.stdout.write(' ')
        if recall < 1.0: sys.stdout.write(' ')
        sys.stdout.write("{:.2%}".format(recall))
        sys.stdout.write("  F1: ")
        if f1 < 0.1: sys.stdout.write(' ')
        if f1 < 1.0: sys.stdout.write(' ')
        sys.stdout.write("{:.2%}".format(f1))
        sys.stdout.write("  #: %d" % gold)
        sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    print("Final Score:")
    prec_micro = 1.0
    if sum(pred_events.values()) > 0:
        prec_micro = float(sum(corrt_events.values())) / float(sum(pred_events.values()))
    recall_micro = 0.0
    if sum(gold_events.values()) > 0:
        recall_micro = float(sum(corrt_events.values())) / float(sum(gold_events.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

## 针对一个句子的评估
def sent_event_eval(pred_events, gold_events, corrt_event_count, gold_event_count, pred_event_count, pred_number_count, gold_number_count):
    count_events(pred_events, pred_event_count, pred_number_count)  # 统计 预测 事件各类型的个数
    count_events(gold_events, gold_event_count, gold_number_count)  # 统计 标注 事件各类型的个数

    pred_simp_events = pred_events.simp_event_set
    pred_bind_events = pred_events.bind_event_set
    pred_regu_events = pred_events.regu_event_set

    gold_simp_events = gold_events.simp_event_set
    gold_bind_events = gold_events.bind_event_set
    gold_regu_events = gold_events.regu_event_set

    # 统计预测正确的事件（九种不同的类型）
    count_sent_events(pred_simp_events, gold_simp_events, corrt_event_count)
    count_sent_events(pred_bind_events, gold_bind_events, corrt_event_count)
    count_sent_events(pred_regu_events, gold_regu_events, corrt_event_count)

def count_sent_events(pred_event_set, gold_event_set, count_corrt_event):
    for pred_event in pred_event_set:
        for gold_event in gold_event_set:
            result = util.event_isequal(pred_event, gold_event)
            if result == True:
                event_type = pred_event.event_type
                assert event_type != ''
                count_corrt_event[event_type] += 1
                # if event_type in constant.SIMP_TYPE:
                #     count_corrt_event['Simple_events'] += 1
                # elif event_type in constant.REGU_TYPE:
                #     count_corrt_event['Complex_event'] += 1
                break

# 统计句子中标注事件各类型的个数以及预测事件各类型的个数
def count_events(sent_events, count_event, complex_argu_count):
    simp_event_set = sent_events.simp_event_set
    bind_event_set = sent_events.bind_event_set
    regu_event_set = sent_events.regu_event_set

    for event in (simp_event_set+bind_event_set+regu_event_set):
        event_type = event.event_type
        count_event[event_type] += 1
        # if event_type in constant.SIMP_TYPE:
        #     count_event['Simple_events'] += 1
        # elif event_type in constant.REGU_TYPE:
        #     count_event['Complex_event'] += 1

            # if event.second_argu == None:
            #     complex_argu_count[event_type+"_one"] += 1
            # else:
            #     complex_argu_count[event_type + "_two"] += 1