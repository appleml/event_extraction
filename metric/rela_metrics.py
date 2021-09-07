from collections import Counter
import sys
from data_prepare.entity import Trigger, Protein
import metric.metric_util as util
import utils.constant as constant

## trigger与argument的关系结果评测
def rela_score(pred_gold_relas):
    corrt_r = Counter()
    gold_r = Counter()
    pred_r = Counter()
    for gold_pred_result in pred_gold_relas: # pred_gold_relas是以句子为单位的，所以可以zip
        gold_ta_pairs, gold_rela_labels, pred_ta_pairs, pred_rela_labels = gold_pred_result
        pred_entity_pairs = list()
        str_pred_pairs = list()
        for pred_pair, pred_label in zip(pred_ta_pairs, pred_rela_labels):
            _, _, trig, argu, _, _ = pred_pair

            trig_type = trig.trig_type
            if pred_label in ["Theme", "Cause"]:
                pred_r[trig_type] += 1
                pred_entity_pairs.append((trig, argu, pred_label))
                if isinstance(argu, Protein):
                    str_pred_pairs.append(trig.trig_idx+"_"+argu.prot_idx+"_"+pred_label)
                elif isinstance(argu, Trigger):
                    str_pred_pairs.append(trig.trig_idx+"_"+argu.trig_idx+"_"+pred_label)
        assert len(set(str_pred_pairs)) == len(str_pred_pairs)

        gold_entity_pairs = list()
        str_gold_pairs = list()
        for gold_pair, gold_label in zip(gold_ta_pairs, gold_rela_labels): ## 包含正例和负例
            _, _, trig, argu, _, _ = gold_pair
            trig_type = trig.trig_type
            if gold_label in ["Theme", "Cause"]:
                gold_r[trig_type] += 1
                gold_entity_pairs.append((trig, argu, gold_label))
                if isinstance(argu, Protein):
                    str_gold_pairs.append(trig.trig_idx+"_"+argu.prot_idx+"_"+gold_label)
                elif isinstance(argu, Trigger):
                    str_gold_pairs.append(trig.trig_idx+"_"+argu.trig_idx+"_"+gold_label)
        assert len(set(str_gold_pairs)) == len(str_gold_pairs)

        # 针对一句话中，计算corrt_r
        for pred_pair in pred_entity_pairs:
            p_trig, p_argu, p_label = pred_pair
            for gold_pair in gold_entity_pairs:
                g_trig, g_argu, g_label = gold_pair
                if isinstance(p_argu, Protein) and isinstance(g_argu, Protein):
                    if util.trig_isequal(p_trig, g_trig) and util.prot_isequal(p_argu, g_argu) and p_label == g_label:
                        corrt_r[p_trig.trig_type] += 1
                        break
                elif isinstance(p_argu, Trigger) and isinstance(g_argu, Trigger):
                    if util.trig_isequal(p_trig, g_trig) and util.trig_isequal(p_argu, g_argu) and p_label == g_label:
                        corrt_r[p_trig.trig_type] += 1
                        break

    ## 计算presion, recall, f_score
    print("Per-relation statistics:")
    relations = gold_r.keys()
    longest_relation = 0
    for relation in sorted(relations):
        longest_relation = max(len(relation), longest_relation)
    for relation in sorted(relations):
        # (compute the score)
        correct = corrt_r[relation]
        guessed = pred_r[relation]
        gold = gold_r[relation]
        prec = 1.0
        if guessed > 0:
            prec = float(correct) / float(guessed)
        recall = 0.0
        if gold > 0:
            recall = float(correct) / float(gold)
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
    if sum(pred_r.values()) > 0:
        prec_micro = float(sum(corrt_r.values())) / float(sum(pred_r.values()))
    recall_micro = 0.0
    if sum(gold_r.values()) > 0:
        recall_micro = float(sum(corrt_r.values())) / float(sum(gold_r.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro