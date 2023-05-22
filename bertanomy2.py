# 脚本ID: 20230516014140
# 注意: 

from transformers import BertModel
from transformers import BertTokenizer
from transformers import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
logging.set_verbosity_error()

bert_model = BertModel.from_pretrained('bert-base-uncased') # 20230515193820
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 20230515193822


'''
[函数注释]
	[功能]
		1. 主要功能: 生成词号(bert_tokenizer认为的第n个单词, 而非空格定界)和wordpiece号的映射关系(字典)
		2. 额外功能
	[设计图]
		1. 索引码:
		2. 文件类型: 
	[参数]
		1. sent
			1. 数据类型: string
			2. 数据结构: string
			3. 参数类型: 必选
			4. 语义: 被处理的句子
			5. 取值范围: 
			6. 获得来源: 手动输入
			7. 样例文件/输入: 
	[用例]
		1. map_pieceid_to_wordid(sent)
			1. 输出
				1. 语义: 生成词号(bert_tokenizer认为的第n个单词, 而非空格定界)和wordpiece号的映射关系(字典)
				2. 数据类型: dict
				3. 数据结构: {0:0,1:1,2:1,3:2,....}
				4. 样例文件/输出: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5, 7: 6}
					4.1 第0个wordpiece对应第0个单词,...,第5,6个wordpiece对应第5个单词...
	[依赖]
		1. 全局对象 bert_tokenizer # 20230515193822
	[已知问题]
		1. [问题1标题]
			1. 问题描述
			2. 问题复现
				1. 复现环境
				2. 复现语句
				3. 复现存档
	[开发计划]
		1. 
		2.
	[备注]
		1. 原理: 先用一个list来承载对照表, list的下标作为wordpiece号, list的元素作为word号, 如果wordpiece以##开头, 则word号不增大(单仍然append进列表中), 如果wordpiece不以##开头, 则word号随wordpiece号+1而+1
'''
def map_pieceid_to_wordid(sent): # 20230515192922
	bert_tokenized_sent = bert_tokenizer(sent, return_tensors='pt')  # 数据类型: <class 'transformers.tokenization_utils_base.BatchEncoding'>; 数据结构: {'input_ids':..., 'token_type_ids':..., 'attention_mask':...}; 参数类型: 必选; 语义: transformers.BertTokenizer输出的对象; 这里只用到它'token_type_ids'这一项; 取值范围: ; 获得来源: bert_tokenized_sent = bert_tokenizer(sent, return_tensors='pt'); 样例文件/输入: 20230515192605.pkl
	'''输入: BertTokenizer(..return_tensors='pt')的返回值\n输出: 字典, 键是wordpiece号, 值是word号'''
	raw_token_ids = bert_tokenized_sent['input_ids'][0] # 数据结构: tensor([  101,  2023,  2003,  1037,  7279,  1037, 28762,  2075,  7279,   102])
	trimmed_raw_token_ids = raw_token_ids[1:-1]
	map_list_wordpos_to_piecepos = []
	cnt = -1
	for tokenid in trimmed_raw_token_ids: #
		if bert_tokenizer.convert_ids_to_tokens([tokenid])[0].startswith('##'):
			cnt = cnt
		else:
			cnt += 1
		map_list_wordpos_to_piecepos.append(cnt)
	return dict(sorted([(i,j) for i,j in enumerate(map_list_wordpos_to_piecepos)]))

'''
[函数注释]
	[功能]
		1. 主要功能: 输入transformers.BertTokenizer的输出, 输出所有wordpieces(文字形式)
		2. 额外功能
	[设计图]
		1. 索引码: 
		2. 文件类型: 
	[参数]
		1. sent
			1. 数据类型: string
			2. 数据结构: string
			3. 参数类型: 必选
			4. 语义: 被处理的句子
			5. 取值范围: 
			6. 获得来源: 手动输入
			7. 样例文件/输入: 
	[用例]
		1. txt_wordpieces(sent)
			2. 输出
				1. 语义: 输出所有wordpieces(文字形式)
				2. 数据类型: list
				3. 数据结构: ['[CLS]','word1','word2',...,'wordn']
				4. 样例文件/输出: ['[CLS]', 'this', 'is', 'a', 'pen', 'a', 'debra', '##ing', 'pen', '[SEP]']
	[依赖]
		1. 全局对象 bert_tokenizer # 20230515193822
	[已知问题]
		1. [问题1标题]
			1. 问题描述
			2. 问题复现
				1. 复现环境
				2. 复现语句
				3. 复现存档
	[开发计划]
		1. 
		2.
	[备注]
		1.
		2. 
'''
def txt_wordpieces(sent): # 20230515195614
	'''输入transformers.BertTokenizer的输出, 输出所有wordpieces(文字形式)'''
	bert_tokenized_sent = bert_tokenizer(sent, return_tensors='pt') # 数据类型: <class 'transformers.tokenization_utils_base.BatchEncoding'>; 数据结构: {'input_ids':..., 'token_type_ids':..., 'attention_mask':...}; 参数类型: 必选; 语义: transformers.BertTokenizer输出的对象; 这里只用到它'token_type_ids'这一项; 取值范围: ; 获得来源: bert_tokenized_sent = bert_tokenizer(sent, return_tensors='pt'); 样例文件/输入: 20230515192605.pkl
	raw_token_ids = bert_tokenized_sent['input_ids'][0] # 数据结构: tensor([  101,  2023,  2003,  1037,  7279,  1037, 28762,  2075,  7279,   102])
	return [bert_tokenizer.convert_ids_to_tokens([i])[0] for i in raw_token_ids]

'''
[函数注释]
	[功能]
		1. 主要功能: 获得一个词在一句话中指定层的词向量
		2. 额外功能
	[设计图]
		1. 索引码: 
		2. 文件类型: 
	[参数]
		1. word
			1. 数据类型: string
			2. 数据结构: string
			3. 参数类型: 必选
			4. 语义: 目标词
			5. 取值范围: 
			6. 获得来源: 手动输入
			7. 样例文件/输入: 
		2. sent
			1. 数据类型: string
			2. 数据结构: string
			3. 参数类型: 必选
			4. 语义: 目标句
			5. 取值范围: 
			6. 获得来源: 手动输入
			7. 样例文件/输入: 
		3. layer
			1. 数据类型: int
			2. 数据结构: int
			3. 参数类型: 必选
			4. 语义: 获取词向量的层
			5. 取值范围: 0-11
			6. 获得来源: 手动输入
			7. 样例文件/输入:
	[用例]
		1. emb_word_in_sent
			1. 输出
				1. 语义: 一个词在一句话中指定层的词向量
				2. 数据类型: list
				3. 数据结构: [tensor(768),tensor(768)....]
				4. 样例文件/输出: 20230516005057.pkl
	[依赖]
		1. 全局对象 bert_tokenizer # 20230515193822
		2. 全局对象 bert_model # 20230515193820

	[已知问题]
		1. [问题1标题]
			1. 问题描述
			2. 问题复现
				1. 复现环境
				2. 复现语句
				3. 复现存档
	[开发计划]
		1. 
		2.
	[备注]
		1. 若一个词在句中有多次出现, 则它
		2. 
'''
def emb_word_in_sent(word,sent,layer): # 20230515195618
	bert_tokenized_sent = bert_tokenizer(sent, return_tensors='pt') # 数据类型: <class 'transformers.tokenization_utils_base.BatchEncoding'>; 数据结构: {'input_ids':..., 'token_type_ids':..., 'attention_mask':...}; 参数类型: 必选; 语义: transformers.BertTokenizer输出的对象; 这里只用到它'token_type_ids'这一项; 取值范围: ; 获得来源: bert_tokenized_sent = bert_tokenizer(sent, return_tensors='pt'); 样例文件/输入: 20230515192605.pkl
	raw_token_ids = bert_tokenized_sent['input_ids'][0] # 数据结构: tensor([  101,  2023,  2003,  1037,  7279,  1037, 28762,  2075,  7279,   102])
	bert_words = [bert_tokenizer.convert_ids_to_tokens([i])[0] for i in raw_token_ids] # 数据结构: ['[CLS]', 'this', 'is', 'a', 'pen', 'a', 'debra', '##ing', 'pen', '[SEP]']
	word_ids = [i for i,j in enumerate(bert_words) if j==word] # 数据结构: [4,8]
	bert_output = bert_model(**bert_tokenized_sent,output_hidden_states=True,output_attentions=True) # 20230515194727 # 数据类型: dict # 数据结构: {key:tensor,key:tensor,key:tuple,key:tuple); 语义: 一句话的所有阶段结果, 详见NOTION: 20230515195127
	return [bert_output['hidden_states'][layer][0][i] for i in word_ids] # 20230516004421 # 数据结构: [tensor, tensor]


'''
[函数注释]
	[功能]
		1. 主要功能: 
		2. 额外功能
	[设计图]
		1. 索引码: 
		2. 文件类型: 
	[参数]
		1. [参数1]
			1. 数据类型: 
			2. 数据结构: 
			3. 参数类型: 
			4. 语义: 
			5. 取值范围: 
			6. 获得来源: 
			7. 样例文件/输入: 
		2. [参数2]
			1. 数据类型: 
			2. 数据结构: 
			3. 参数类型: 
			4. 语义: 
			5. 取值范围: 
			6. 获得来源: 
			7. 样例文件/输入: 
	[用例]
		1. [用例1]
			1. 语句: 
			2. 输出
				1. 语义: 
				2. 数据类型: 
				3. 数据结构: 
				4. 样例文件/输出: 
	[依赖]
		1. 
		2. 
	[已知问题]
		1. [问题1标题]
			1. 问题描述
			2. 问题复现
				1. 复现环境
				2. 复现语句
				3. 复现存档
	[开发计划]
		1. 
		2.
	[备注]
		1.
		2. 
'''
def viz_scatter_tsne(embs,randstate):
	def dim_reduce(embs,randstate):
	# 将300维的词向量降维2维
		ndim = 2
		# 实例化TSNE器
		tsne = TSNE(n_components=ndim,random_state=randstate)
		# 降维
		dim_reduced_vecs = tsne.fit_transform(embs)
		# 降维后的二维向量的横坐标
		x_vals = [v[0] for v in dim_reduced_vecs]
		# 降维后的二维向量的纵坐标
		y_vals = [v[1] for v in dim_reduced_vecs]
		return x_vals,y_vals
	embs = [i.detach().numpy() for i in embs]
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.scatter(*dim_reduce(embs,randstate))
	plt.show()
	return


def tmp_word_sense_cloud(word,corpus_path,bert_layer,tsne_randstate):
	sents = []
	embs = []
	corp = open(corpus_path,mode='r',encoding='utf-8')
	while True:
		try:
			sent = next(corp)
			if f' {word} ' in sent:
				sents.append(sent)
		except StopIteration:
			break
	for s in sents:
		embs.append(emb_word_in_sent(word, s, bert_layer)[0])
	viz_scatter_tsne(embs, tsne_randstate)
	return