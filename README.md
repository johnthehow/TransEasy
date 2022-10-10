# BERTET: BERT Explainability Toolkit
## glassbox_bert
### 功能概述
1. 在输入一句话后, 得到一个封装了数据和模型的实例
2. 不同于原生Huggingface BERT, glassbox_bert能够方便地返回所有能够想想得到的中间阶段输出, 而不仅仅是Attention和Hidden States
3. 实例和Vaswani2017的概念图高度对应, 方便不了解Huggingface技术细节者操作
4. 中间阶段数据被分为6组
	4.1 Symbolic组: 该组变量前缀均为 sym_
	4.2 Pre-Embedding组: 该组变量前缀均为 preemb_
	4.3 Multi-Head Self-Attention组: 该组变量前缀均为 selfattn_
	4.4 Add & Norm 1组: 该组变量前缀均为 addnorm1_
	4.5 Feedforward组: 该组变量前缀均为 ffnn_
	4.6 Add & Norm 2组: 该组变量前缀均为 addnorm2_
