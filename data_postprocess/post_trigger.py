# 如果当前词是protein, 则一定不会是trigger
# gold_prot_sequence 是标注的protein序列
# pred_trig_sequence 是预测的trigger序列
TriggerType = ['Gene_expression', 'Transcription', 'Phosphorylation', 'Protein_catabolism', 'Localization', 'Binding', 'Regulation', 'Positive_regulation', 'Negative_regulation']
ProtType = ['B-Protein', 'I-Protein', 'E-Protein', 'S-Protein']
def trig_regularization(trig2id, gold_prot_sequence, pred_trig_sequence):
    regu_trig_labels = list()
    for gold_prot_tag, pred_trig_tag in zip(gold_prot_sequence, pred_trig_sequence):
        if gold_prot_tag in ProtType and pred_trig_tag in TriggerType:
            regu_trig_labels.append('Other')
        else:
            regu_trig_labels.append(pred_trig_tag)
    assert len(regu_trig_labels) == len(pred_trig_sequence)
    regu_trig_ids = [trig2id[t] for t in regu_trig_labels]
    return regu_trig_labels, regu_trig_ids