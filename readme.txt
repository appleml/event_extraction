2020年11月11日
项目名称: gcn-deprela-bilstm-protrig-gcn-relabind-BIO-two: 在评测的过程中将simple 和 complex去掉了,怕影响最终的评测结果

2020年11月17日
(1) trig-argument对由一次性处理转换成一个预测一对
(2) bind argument也做同样的处理

2020年11月22日
目前作为备份在使用
四个子任务共用bilstm, binding argument和 relation extraction继续共用了gcn, 这对于binding argument有突出贡献
在train中有700多个跨句子事件, devel有300多个跨句子事件,需要对跨句子事件进行处理

2020年12月18号
目前版本的框架模型时: 四个子任务
(1) join protein的识别
(2) trigger的识别
(3) trigger与argument关系之间的判断
(4) binding的论元问题
其中四个子任务共用了bilstm, 其中任务(3)和任务(4)在共享bilstm的基础上又共用了gcn

2020年12月19日
Share_bilstm 之前的做法是分别在jprot_model.py, trig_model.py以及gcn.py中分别都实例化了, 目前的改正方法是是在trainer.py中实例化了一次, 再查看效果

2020年12月24号
增加了cross-sentence信息

2021年2月12日:
按照范创和立波说的修改了学习率以及其他, 目前是是四个模块是联合的,效果不好,放服务器备份

2021年3月8号:
jprot_lr和trig_lr更改为1e-04(原先是1e-03)
cjprot_dim和ctrig_dim原来是45, 更改为30
