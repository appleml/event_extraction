## 为了对比预测出的trigger的全局位置是否正确，根据trigger的全局位置是否能做原文中找到
import os
def find_trigger(origin_context_path, genert_trig_path):
    files = os.listdir(genert_trig_path)
    for file_name in files:
        if file_name == ".a2":
            continue
        dot_locate = file_name.rfind(".")
        pure_name = file_name[0:dot_locate]
        with open(origin_context_path+"/"+pure_name+".txt", "r") as context_fp, open(genert_trig_path+"/"+file_name, "r") as trig_fp:
            context = context_fp.read()
            for trig_line in trig_fp:
                if trig_line.startswith("T"):
                    trig_info = trig_line.replace("\t", " ").split()
                    trig_cstart = int(trig_info[2])
                    trig_cend = int(trig_info[3])
                    trig_name = trig_info[4]
                    context_word = context[trig_cstart:trig_cend]
                    if trig_name != context_word:
                        print(file_name)
                        print(trig_line)
                        print(context_word)

if __name__ == "__main__":
    origin_devel_path = "/home/fang/Downloads/BioNLP-ST_2011_genia_tools_rev1/2011_devel_data"
    gen_devel_trig_path = "/home/fang/Downloads/BioNLP-ST_2011_genia_tools_rev1/output_a2"
    find_trigger(origin_devel_path, gen_devel_trig_path)