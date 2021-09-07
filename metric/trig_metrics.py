from collections import Counter
import sys
import utils.constant as constant

def eval_trigger(pred_seqs, gold_seqs):
    corrt_res = Counter()
    gold_res = Counter()
    pred_res = Counter()
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs):
        for pred_label, gold_label in zip(pred_seq, gold_seq):
            if pred_label == gold_label and gold_label != "Other":
                corrt_res[gold_label] += 1

            if gold_label != "Other":
                gold_res[gold_label] += 1

            if pred_label != "Other":
                pred_res[pred_label] += 1

    ## 计算presion, recall, f_score
    print("Per-relation statistics:")
    relations = gold_res.keys()
    longest_relation = 0
    for relation in sorted(relations):
        longest_relation = max(len(relation), longest_relation)
    for relation in sorted(relations):
        # (compute the score)
        correct = corrt_res[relation]
        guessed = pred_res[relation]
        gold = gold_res[relation]
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
    if sum(pred_res.values()) > 0:
        prec_micro = float(sum(corrt_res.values())) / float(sum(pred_res.values()))
    recall_micro = 0.0
    if sum(gold_res.values()) > 0:
        recall_micro = float(sum(corrt_res.values())) / float(sum(gold_res.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro