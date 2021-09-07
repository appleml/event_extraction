import sys
from collections import Counter
import utils.constant as constant
## input as sentence level labels
def eval_jprot(predict_lists, golden_lists):
    sent_num = len(golden_lists)
    golden_global = Counter()
    predict_global = Counter()
    right_global = Counter()

    for idx in range(0,sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]

        gold_matrix = get_ner_BIO(golden_list)
        pred_matrix = get_ner_BIO(predict_list)

        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        count_type_number(gold_matrix, golden_global)
        count_type_number(pred_matrix, predict_global)
        count_type_number(right_ner, right_global)

    ## printing
    print("Per-relation statistics:")
    relations = golden_global.keys()
    longest_relation = 0
    for relation in sorted(relations):
        longest_relation = max(len(relation), longest_relation)
    for relation in sorted(relations):
        # (compute the score)
        correct = right_global[relation]
        guessed = predict_global[relation]
        gold = golden_global[relation]
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
    if sum(predict_global.values()) > 0:
        prec_micro = float(sum(right_global.values())) / float(sum(predict_global.values()))
    recall_micro = 0.0
    if sum(golden_global.values()) > 0:
        recall_micro = float(sum(right_global.values())) / float(sum(golden_global.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

## 标签系统是BIO
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
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

def count_type_number(label_list, count_num):
    for label in label_list:
        market = label.index(']')
        type = label[market+1:].lower().capitalize()
        count_num[type] += 1
        if type in constant.SIMP_TYPE:
            count_num['simple'] += 1
        elif type in constant.REGU_TYPE:
            count_num['complex'] += 1