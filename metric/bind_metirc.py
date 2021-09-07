from collections import Counter
import sys
from metric.metric_util import trig_isequal, prot_isequal
## bind_exam_res = [(gold_pair_bind, gold_pair_seq, pred_pair_bind, pred_bind_res)]
## gold_pair = (trig_idx+"&"+argu2.prot_idx+"&"+argu1.prot_idx, trig1, argu2, label2, argu1, label1)
def bind_score(bind_exam_res):
    corrt_b = Counter()
    gold_b = Counter()
    pred_b = Counter()
    for gold_pair_bind, gold_pair_seq, pred_pair_bind, pred_bind_seq in bind_exam_res: #对句子进行
        gold_pairs = list()
        for gold_pair, gold_rela in zip(gold_pair_bind, gold_pair_seq): #对句子中的集合进行遍历
            if gold_rela == 1:
                gold_b["bind"] += 1
                trig1, argu1, label1, argu2, label2 = gold_pair
                assert label1 == "Theme" and label2 == "Theme"
                gold_pairs.append((trig1, argu1, argu2))
        pred_pairs = list()
        for pred_pair, pred_rela in zip(pred_pair_bind, pred_bind_seq): #对句子中的集合进行遍历
            if pred_rela == 1:
                pred_b["bind"] += 1
                trig1, argu1, label1, argu2, label2 = pred_pair
                assert label1 == "Theme" and label2 == "Theme"
                pred_pairs.append((trig1, argu1, argu2))
        ## 同一个句子中计算 corrt_b
        for pred_exam in pred_pairs:
            p_trig, p_argu1, p_argu2 = pred_exam
            for gold_exam in gold_pairs:
                g_trig, g_argu1, g_argu2 = gold_exam
                if trig_isequal(p_trig, g_trig) and prot_isequal(p_argu1, g_argu1) and prot_isequal(p_argu2, g_argu2):
                    corrt_b['bind'] += 1
                elif trig_isequal(p_trig, g_trig) and prot_isequal(p_argu1, g_argu2) and prot_isequal(p_argu2, g_argu1):
                    corrt_b['bind'] += 1

    # 结果评测
    # Print verbose information
    print("binding statistics:")
    relations = gold_b.keys()
    longest_relation = 0
    for relation in sorted(relations):
        longest_relation = max(len(relation), longest_relation)
    for relation in sorted(relations):
        # (compute the score)
        corrt = corrt_b[relation]
        pred = pred_b[relation]
        gold = gold_b[relation]
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
    if sum(pred_b.values()) > 0:
        prec_micro = float(sum(corrt_b.values())) / float(sum(pred_b.values()))
    recall_micro = 0.0
    if sum(gold_b.values()) > 0:
        recall_micro = float(sum(corrt_b.values())) / float(sum(gold_b.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro