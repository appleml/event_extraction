## 以文件为单位写出
## file = [sent_info]
## 以下是问题：trigger的id的问题，应该记录上一句中trigger的idx, 下一句的idx 应该紧接上一句的 idx
## ## trigger 位置已经修正，不在此处修正，只在此处写出
## 仅仅只是给出了trig_start, 和trig_end, 需要计算出(trig_oldchar_start, trig_oldchar_end, trig_newchar_start, trig_newchar_end)
def write_event(output_path, file_trigs, file_events):
    for file_name, trig_set in file_trigs.items():
        dot_position = file_name.rfind(".")
        pure_name = file_name[0:dot_position]
        with open(output_path + "/" + pure_name + ".a2", "w") as output_fp:
            for trig in trig_set:
                str_trig = trig2str(trig)
                output_fp.write(str_trig + "\n")
                output_fp.flush()
            if file_name in file_events.keys():
                event_set = file_events[file_name]
                for evet in event_set:
                    str_event = event2str(evet)
                    output_fp.write(str_event + "\n")
                    output_fp.flush()

'''
写出的形式：T18	Negative_regulation 0 15	Down-regulation
'''
def trig2str(trig):
    trig_idx = trig.trig_idx
    type_posit = trig.trig_type + " " + str(trig.trig_context_start) + " " + str(trig.trig_context_end)
    trig_name = trig.trig_name
    return trig_idx+"\t"+type_posit+"\t"+trig_name

## E1	Negative_regulation:T18 Theme:E2
def event2str(evet):
    event_idx = evet.event_idx
    event_trig_info = evet.event_type+":"+evet.event_trig_idx
    assert evet.event_trig_idx != ""
    first_argu_info = evet.first_argu_type+":"+evet.first_argu_idx
    assert evet.first_argu_idx != ""
    if evet.second_argu != None:
        if evet.second_argu_type == "Theme":
            second_argu_info = "Theme2:"+evet.second_argu_idx
        else:
            assert evet.second_argu_type == "Cause"
            second_argu_info = evet.second_argu_type + ":" + evet.second_argu_idx
        return event_idx+"\t"+ event_trig_info+" "+first_argu_info+" "+second_argu_info
    else:
        return event_idx+"\t"+ event_trig_info+" "+first_argu_info
