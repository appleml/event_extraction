import os
## 将 genia 生成的结果与 parser 的结果进行合并，这样可以避免不一致
## parse_line: 1	Down-regulation	_	NN	_	_	0	root	_	_
## genia_line: Down-regulation	NN	B-NP	O	Other	Other	Negative_regulation
def merge_data(genia_path, parser_path, genia_parser):
    files = os.listdir(genia_path)
    for file_name in files:
        with open(genia_path + "/" + file_name, "r") as genia_fp, open(parser_path + "/" + file_name, "r") as parser_fp, open(genia_parser+"/"+file_name, 'w') as write_data:
            for genia_line, parse_line in zip(genia_fp, parser_fp):
                genia_line = genia_line.strip().strip("\n")
                parse_line = parse_line.strip().strip("\n")
                genia_info = genia_line.split('\t')
                parse_info = parse_line.split('\t')
                if not len(genia_line) and not len(parse_line):
                    write_data.write("\n")
                    write_data.flush()
                else:
                    if genia_info[0] != parse_info[1]:
                        print(file_name)
                        print(genia_info)
                        print(parse_line)
                        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                        break
                    write_data.write(genia_line+"\t###\t"+parse_line+"\n")
                    write_data.flush()


if __name__ == '__main__':
    train_genia = "/home/fang/myworks/gcn-trees-origin-unbert/dataset/2011_genia/train_genia"
    train_parser = "/home/fang/myworks/gcn-trees-origin-unbert/dataset/2011_parser/train_data"
    train_genia_parser = "/home/fang/myworks/gcn-trees-origin-unbert/dataset/2011_genia_parser/train_data"
    merge_data(train_genia, train_parser, train_genia_parser)

    devel_genia = "/home/fang/myworks/gcn-trees-origin-unbert/dataset/2011_genia/devel_genia"
    devel_parser = "/home/fang/myworks/gcn-trees-origin-unbert/dataset/2011_parser/devel_data"
    devel_genia_parser = "/home/fang/myworks/gcn-trees-origin-unbert/dataset/2011_genia_parser/devel_data"
    merge_data(devel_genia, devel_parser, devel_genia_parser)
