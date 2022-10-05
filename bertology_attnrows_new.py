import torch
import os
from transformers import BertModel
from transformers import BertTokenizer
from transformers import logging
from pathlib import Path
import pickle

logging.set_verbosity_error()

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 函数ID: 20220403124623
# 从指定语料库 返回 包含目标词的 句子们
def space_len_sents_selector(word,sent_length,corpus):
	'''从指定语料库 返回 包含目标词的 句子们'''
	sents_len = [sent for sent in corpus if len(sent) == sent_length]
	sents_len_word = [' '.join(sent) for sent in sents_len if word in sent]
	return sents_len_word

# 功能: 对一句话进行多个层面的分词
# 函数ID: 20220403124640
# 输入: 一句话(一个字符串)
# 输出: 字典
# 	tokenized_sent: 字典: BERT_tokenizer原始输出: raw_input_ids, token_type_ids, attention_mask
# 	raw_input_ids: 列表: 原始分词ID表: 未削去两端的,未经重新合并的, raw_input_ids的 列表
# 	trimmed_input_ids: 列表: 削尾分词ID表: 削去两端的,未经重新合并的, raw_input_ids的, 列表
# 	raw_tokens: 列表: 削尾wordpiece分词句： 包括 [CLS]和[SEP]的, wordpiece(字符串)的列表
# 	dense_tokens: 列表: 空格分词句：  削去两端的, 重新合并的, wordpiece(字符串)的列表, 相当于空格分词
# 	idmap_list: 列表: 重新合并映射表: 重新合并token所需的映射表, 元素为int
# 	idmap_dict: 字典: 重新合并映射表: 重新合并token所需的映射表, 键为int, 值为int
# 依赖
# 	全局对象bert_tokenizer
def complete_tokenizer(sent):
	'''输入一句话(字符串), 返回六种形式表达的BERT_tokenization的结果'''
	def get_id_map(trimmed_inputids):
		# 获得token-word合并映射表
		# 获得token-word合并映射表:list版
		cnt = -1
		idmap_list = [] 
		for tokenid in trimmed_inputids: # token-word合并映射表 list
			if bert_tokenizer.convert_ids_to_tokens([tokenid])[0].startswith('##'):
				cnt = cnt
			else:
				cnt += 1
			idmap_list.append(cnt)
		# 获得token-word合并映射表:dict版
		idmap_dict = dict()
		for wordid in idmap_list:
			idmap_dict[wordid] = [tokenid for tokenid,realwordid in enumerate(idmap_list) if realwordid==wordid]
		return {'idmap_list': idmap_list, 'idmap_dict': idmap_dict}

	# 重新合并wordpiece分词后的句子
	def get_onepiece_tokens(tokenlist,maplist):
		mapdict = dict()
		for i in maplist:
			mapdict[i] = [k for k,v in enumerate(maplist) if v==i]
		tokendict = dict()
		for k in mapdict:
			tokendict[k] = [tokenlist[token] for token in mapdict[k]]
		tokendictj = dict()
		for item in tokendict.items():
			raw_tokens = item[1]
			new_tokens = []
			for j in item[1]:
				if j.startswith('##'):
					new_tokens.append(j[2:])
				else:
					new_tokens.append(j)
			tokendictj[item[0]] = ''.join(new_tokens)
		return tokendictj
	
	# tokenized句子
	tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
	# 获得tokenized句子的input_id串 '101 2065 3999 2837 ... 102'
	raw_input_ids = tokenized_sent['input_ids'][0]
	# 截掉句子input_id两端的[101] 和 [102]
	trimmed_input_ids = raw_input_ids[1:-1]
	# wordpiece合并映射表
	idmap = get_id_map(trimmed_input_ids)
	idmap_list = idmap['idmap_list']
	idmap_dict = idmap['idmap_dict']
	# 截掉句子两端[CLS]和[SEP]后的tokenization结果
	raw_input_tokens = bert_tokenizer.convert_ids_to_tokens(raw_input_ids)
	trimmed_input_tokens = raw_input_tokens[1:-1]
	dense_input_tokens_dict = get_onepiece_tokens(trimmed_input_tokens,idmap_list)
	dense_input_tokens_list = list(dense_input_tokens_dict.values())
	return {'tokenized_sent':tokenized_sent,'raw_input_ids':raw_input_ids,'trimmed_input_ids':trimmed_input_ids,'raw_tokens':raw_input_tokens,'trimmed_tokens':trimmed_input_tokens,'dense_tokens': dense_input_tokens_list,'idmap_list':idmap_list,'idmap_dict':idmap_dict}

# 功能: 删除attention矩阵中[CLS]和[SEP]对应的行和列, 并将剩余的值相应放大
# 函数ID: 20220403124654
# 输入语义: Attention矩阵
# 输入数据: numpy矩阵
# 输入来源: 
# 输出:
# 	数据类型: 矩阵
# 	语义: 削边放大矩阵
# 依赖:
# 	无
# 被依赖:
# 	attn_matrix_denser
# 	
def attn_matrix_trim_scale(attn_matrix):
	''' 删除attention矩阵中[CLS]和[SEP]对应的行和列, 并将剩余的值相应放大'''
	core_matrix = attn_matrix[1:-1,1:-1]
	container_tensor = torch.zeros(core_matrix.shape)
	row_cnt = 0
	for row in core_matrix:
		mtplr = 1/sum(row)
		scaled_row = row*mtplr
		container_tensor[row_cnt] = scaled_row
		row_cnt += 1
	return container_tensor

# 功能:	把wordpiece的token为行列的 削皮的 attention矩阵, 合并成原词为行列的削皮矩阵
# 函数ID: 20220403124737
# 输入来源: 
# 	idmap_dict: complete_tokenizer()
# 	trim_scale_attn_matrix: attn_matrix_trim_scale()
# 输入数据类型:
	# idmap_dict: 字典
	# trim_scale_attn_matrix: 矩阵
# 被依赖: 
def attn_matrix_denser(idmap_dict,trim_scale_attn_matrix):
	# 未削皮放缩的矩阵, 每一行之和为1, 每一列之和不为1
	# 削皮放缩后的矩阵, 每一行之和为1, 每一列之和不为1
	# 先把行压缩
	def rebuild_row(idmap_dict,trim_scale_attn_matrix):
		# 新建一个承载变量, 行数是压缩后的行数, 列数是删除掉[SEP]和[CLS]后的seq长度(>=压缩后的行数)
		new_tensor = torch.zeros((len(idmap_dict.keys()),len(trim_scale_attn_matrix)))
		# 对于压缩seq的每个词
		for row_no_new in idmap_dict.keys():
			# 承载变量中每个词(而非wordpiece)对应的行 是 削皮矩阵的中该词对应的多行 多行对位相加成一行 的行
			new_tensor[row_no_new] = trim_scale_attn_matrix[idmap_dict[row_no_new]].sum(axis=0)/len(idmap_dict[row_no_new])
		return new_tensor
	# 再把列压缩
	def rebuild_col(idmap_dict,trim_scale_attn_matrix):
		new_tensor = torch.zeros((len(idmap_dict.keys()),len(idmap_dict.keys())))
		for col_no_new in idmap_dict:
			new_tensor[:,col_no_new] = trim_scale_attn_matrix[:,idmap_dict[col_no_new]].sum(axis=1)
		return new_tensor
	row_proced = rebuild_row(idmap_dict,trim_scale_attn_matrix)
	col_proced = rebuild_col(idmap_dict,row_proced)
	matrix_proced = col_proced
	return matrix_proced

def get_word_position(word,sent):
	'''获得一个词在句中的位置(dense attn matrix的行号)'''
	# tokenized句子
	tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
	# 获得tokenized句子的input_id串 '101 2065 3999 2837 ... 102'
	input_ids = tokenized_sent['input_ids'][0]
	# 截掉句子input_id两端的[101] 和 [102]
	input_ids = input_ids[1:-1]

	# 获得token-word合并映射表
	better_token = complete_tokenizer(sent)
	token_word_id_map_list = better_token['idmap_list']
	token_word_id_map_dict = better_token['idmap_dict']
	
	# 获得目标词在BERT词表中的编号
	target_word_id = bert_tokenizer.convert_tokens_to_ids(word)
	# 获得目标词在token-word合并映射表转换前的attention矩阵中的行号
	target_word_attn_row_num_token =  (input_ids == target_word_id).nonzero()[0]
	# 获得目标词在token-word合并映射表转换后的attention矩阵中的行号
	target_word_attn_row_num_dense = token_word_id_map_list[target_word_attn_row_num_token]

	return target_word_attn_row_num_dense

def concat_len_sents_selector(word,sent_len,corpus,sent_max):
	# 筛选BERT分词后,长度等于sent_len的句子
	# 因为BERT_TOKENIZER并不按照空格分词, 在sent_selector中预选长度为n的句子, 在BERT分词后, 长度可能大于n
	# 所以, 从sent_selector中选取长度小于sent_len的句子, 从中会产生BERT分词后, 长度恰好等于sent_len的句子
	# limit限制最多保存多少个句子, 文中取16000
	sents = space_len_sents_selector(word,sent_len,corpus)
	sents += space_len_sents_selector(word,sent_len-1,corpus)
	sents += space_len_sents_selector(word,sent_len-2,corpus)
	sents += space_len_sents_selector(word,sent_len-3,corpus)
	truelen_sents = []
	limit_cnt = 1
	for sent in sents:
		if limit_cnt <=sent_max:
			# tokenized句子
			res_better_token  = complete_tokenizer(sent)
			true_len  = len(res_better_token['dense_tokens'])
			if true_len == sent_len:
				truelen_sents.append(sent)
				limit_cnt += 1
		else:
			break
	return truelen_sents

# 功能: 主函数单词版, 获取词在句中对应的attn行和位置编号
# 函数ID: 20220403124532
# 参数来源:
#	word: 键入字符串
# 	sent_len: 键入整数
# 	corpus: pickle加载文件(来源LENOVO-PC: 20220403160854)
# 	sent_max: 键入整数
# 依赖函数:
# 	true_length_sents_selctor
# 	bert_tokenizer
# 	get_word_position
# 	attn_matrix_trim_scale
# 	attn_matrix_denser
# 输出:
# 	tensor.shape = 12, 12, sentno, 17
def get_word_attn_rowlabs(word,sent_len,corpus,sent_max):
	# 根据真正长度等于sent_len的句子, 提取它们中, 各个目标词各自的144行长度为sent_len的attention行
	word_pos = []
	truelen_sents = concat_len_sents_selector(word,sent_len,corpus,sent_max)
	len_truelen_sents = len(truelen_sents)
	# 创建承载容器
	attn_rowlabs = torch.zeros(12,12,len_truelen_sents,sent_len+1)
	# 获得attention行 和 词在句中位置的标签
	# tsent = truelength_sent
	tsent_cnt = 0
	for tsent in truelen_sents:
		tokenized_sent = bert_tokenizer(tsent,return_tensors='pt')
		res_better_token  = complete_tokenizer(tsent)
		idmap_dict = res_better_token['idmap_dict']
		idmap_list = res_better_token['idmap_list']
		sent_attn_144 = torch.stack(bert_model(**tokenized_sent,output_attentions=True)['attentions']).squeeze()
		word_row_num = get_word_position(word,tsent)
		word_pos.append(word_row_num)
		for layer in range(12):
			for head in range(12):
				attn_trimmed = attn_matrix_trim_scale(sent_attn_144[layer][head])
				attn_dense = attn_matrix_denser(idmap_dict,attn_trimmed)
				attn_row = attn_dense[word_row_num].detach()
				attn_rowlabs[layer][head][tsent_cnt][:-1]=attn_row
				attn_rowlabs[layer][head][tsent_cnt][-1] = word_row_num
		tsent_cnt += 1
	return attn_rowlabs

# 功能: 主函数(单词列表版)
# 	从清理后的语料库获取词的attention行和词在句中位置的标签, 用于probe训练
# 函数ID: 20220403162134
# 依赖函数:
# 	get_word_attn_rowlabs 20220403124532
# 参数解释:
# 	wordlist: 键入
# 	sent_len: 句长, 整数, 键入
# 	corpus: pickle加载文件(来源LENOVO-PC: 20220403160854)
# 	sent_max: 句子数上限, 整数, 键入
# 输出:
# 	pkl文件
# 		tensor.shape = 所有单词在指定句长出现的句子数, 句长+1
# 前置语句:
# 	with open(r'D:\thehow\3\POSDIST\CORPUS\2_CORPUS\EN\CODENRES\RES\CAT\en_lepzig_all.pkl',mode='rb') as pkl:
# 		corpus_en = pickle.load(pkl)
# 示例语句:
# 	get_words_attn_rowlabs(['a','an','about','after'],16,corpus_en,16000,r'd:/temp/')
# 快速实验方法:
# 	1. corpus使用较小的
# 	2. wordlist使用较少的
# 	*. 10k句和三个词的实验用时约8分钟
def get_words_attn_rowlabs(wordlist,sent_len,corpus,sent_max,save_path):
	# 计算每一个词的attn行和句中位置标签lab, 作为tensor放置于容器indiv_word_attnrowlabs144中
	words_attnrowlabs144 = []
	words_line_cnt = []
	for word in wordlist:
		oneword_attnrowlabs = get_word_attn_rowlabs(word,sent_len,corpus,sent_max)
		words_attnrowlabs144.append(oneword_attnrowlabs)
		word_line_cnt = oneword_attnrowlabs.shape[2]
		words_line_cnt.append(word_line_cnt)
		print(f'Attention weights and position for word {word} in {word_line_cnt} length-{sent_len} sentences acquired.')

	# 创建结果容器
	words_line_cnt_sum = sum(words_line_cnt)
	heads_attnrowlabs = torch.zeros(12,12,words_line_cnt_sum,sent_len+1)
	for layer in range(12):
		for head in range(12):
			onehead_attnrowlabs = []
			for attnrowlabs144 in words_attnrowlabs144:
				onehead_attnrowlabs.append(attnrowlabs144[layer][head])
			onehead_attnrowlabs = torch.cat(onehead_attnrowlabs)
			heads_attnrowlabs[layer][head] = onehead_attnrowlabs
			filename = Path(save_path).joinpath(f'{layer+1:02d}_{head+1:02d}').joinpath('data').joinpath(f'attnrowlabs_{layer+1:02d}_{head+1:02d}.pkl')
			os.makedirs(Path(save_path).joinpath(f'{layer+1:02d}_{head+1:02d}').joinpath('data'),exist_ok=False)
			with open(filename,mode='wb') as pkl:
				pickle.dump(onehead_attnrowlabs,pkl)
	return heads_attnrowlabs