
import operator
'''
前提条件: token必须是protein, 才能是 join protein
比较gold sequence 和 pred sequence
(1) 如果gold tag 是 Other, pred tag是其他标签的情况下将pred tag修改为other
(2) 如果gold tag 是标注的B-protein, I-protein, E-protein, S-protein, pred tag如果标记为 Other 则不修改, 否则就要修改

 (1) 如果pred_tag的B-Protein,E-Protein和gold_tag一致, 则将pred_tag与gold_tag保持一致
 (2) 其他情况下将pred_tag都改为other
 标签系统用得BIOES
'''
def jprot_regularization(jprot2id, gold_labels, pred_labels):
    regu_pred_labels = list()
    gold_protein_fragment = list()
    pred_protein_fragment = list()
    # print('-----------------第一行是gold, 第二行是预测标签序列, 第三行是规范化后的序列------------------')
    # print(" ".join(gold_labels))
    # print(" ".join(pred_labels))
    for gold_tag, pred_tag in zip(gold_labels, pred_labels):
        if gold_tag == 'S-Protein':
            if pred_tag == 'S-Protein':
                regu_pred_labels.append('S-Protein')
            elif pred_tag != 'S-Protein': ##一定出错啦
                regu_pred_labels.append('Other')

        elif gold_tag == 'B-Protein':  ## 两种可能(1) 将预测全部改为other, (2)将这一段按gold tag 序列走
            gold_protein_fragment.append(gold_tag)
            pred_protein_fragment.append(pred_tag)

        elif gold_tag == 'I-Protein':
            gold_protein_fragment.append(gold_tag)
            pred_protein_fragment.append(pred_tag)

        elif gold_tag == 'E-Protein':
            gold_protein_fragment.append(gold_tag)
            pred_protein_fragment.append(pred_tag)

            if operator.eq(gold_protein_fragment,pred_protein_fragment):
                regu_pred_labels.extend(gold_protein_fragment)

            elif gold_protein_fragment[0] == pred_protein_fragment[0] and gold_protein_fragment[-1] == pred_protein_fragment[-1]:
                regu_pred_labels.extend(gold_protein_fragment)
            else:
                regu_pred_labels.extend(['Other']*len(gold_protein_fragment))


            gold_protein_fragment.clear()
            pred_protein_fragment.clear()

        elif gold_tag == 'Other': ##该条件一定要放到最后
            regu_pred_labels.append('Other')
    print(" ".join(gold_labels))
    print(" ".join(regu_pred_labels))

    assert len(regu_pred_labels) == len(gold_labels)
    regu_jprot_ids = [jprot2id[jp] for jp in regu_pred_labels]
    return regu_pred_labels, regu_jprot_ids

## 标签系统是BIO
def jprot_regularization_two(jprot2id, gold_labels, pred_labels):
    regu_pred_labels = list()
    gold_protein_fragment = list()
    pred_protein_fragment = list()

    for gold_tag, pred_tag in zip(gold_labels, pred_labels):
        if gold_tag == 'B-Protein':  ## 两种可能(1) 将预测全部改为other, (2)将这一段按gold tag 序列走
            gold_protein_fragment.append(gold_tag)
            pred_protein_fragment.append(pred_tag)

        elif gold_tag == 'I-Protein':
            gold_protein_fragment.append(gold_tag)
            pred_protein_fragment.append(pred_tag)

        elif gold_tag == 'Other': ##该条件一定要放到最后
            if len(gold_protein_fragment) != 0 and len(pred_protein_fragment) != 0:
                if gold_protein_fragment[0] == pred_protein_fragment[0] and gold_protein_fragment[-1] == pred_protein_fragment[-1]:
                    regu_pred_labels.extend(gold_protein_fragment)
                else:
                    regu_pred_labels.extend(['Other']*len(gold_protein_fragment))

                gold_protein_fragment.clear()
                pred_protein_fragment.clear()

            else:
                regu_pred_labels.append('Other')

    assert len(regu_pred_labels) == len(gold_labels)
    regu_jprot_ids = [jprot2id[jp] for jp in regu_pred_labels]
    return regu_pred_labels, regu_jprot_ids