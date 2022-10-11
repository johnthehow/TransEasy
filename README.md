# BERTET: BERT Explainability Toolkit
## glassbox_bert
### 功能概述
1. 在输入一句话后, glassbox_bert类返回一个封装了数据和模型的实例
2. 不同于原生Huggingface BERT, glassbox_bert能够 方便地返回 所有 能够想想得到的 中间阶段输出, 而不仅仅是Attention和Hidden States
3. glassbox_bert实例 和 Vaswani2017的概念图 高度对应, 方便不了解Huggingface技术细节者操作
4. 中间阶段数据被分为6组
	4.1 Symbolic组: 该组变量前缀均为 sym_
	4.2 Pre-Embedding组: 该组变量前缀均为 preemb_
	4.3 Multi-Head Self-Attention组: 该组变量前缀均为 selfattn_
	4.4 Add & Norm 1组: 该组变量前缀均为 addnorm1_
	4.5 Feedforward组: 该组变量前缀均为 ffnn_
	4.6 Add & Norm 2组: 该组变量前缀均为 addnorm2_
5. 自动产生的结果为Pipeline组: 该组变量前缀均为 pipeline_
6. 手动产生的hidden_states结果(对应pipeline_hiddens)为 manual_hiddens
### 注意事项
1. query,key,value权重矩阵中, 每64行对应一个head, bias向量中, 每64个对应一个head
2. 矩阵乘法都是XW.T而不是反过来
3. FFNN组的激活函数是gelu
4. Multi-Head Self-Attention组的Softmax轴向是-1
