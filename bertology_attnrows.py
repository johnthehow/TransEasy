# 同步测试
import math
import torch
import numpy
import random
import matplotlib.pyplot as plt
from torch import nn
from transformers import BertModel
from transformers import BertTokenizer
from transformers.activations import gelu
from transformers import logging
from sklearn.manifold import TSNE
from pathlib import Path
import pickle
from thehow import t

logging.set_verbosity_error()

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 函数ID: 20220403124623
# 从指定语料库 返回 包含目标词的 句子们
def fix_len_sent_extractor(word,sent_length,corpus):
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

# 函数id: 20220403130326
def get_word_attnpos(word:str,sent:str,attn_layer:int, attn_head:int):
	''' 返回指定词在指定句子的指定head的关注位置分布, 返回值有6项(tuple)
	返回值0: 指定词在指定句子的指定head的关注位置分布
	返回值1: WordPiece-常规分词映射表
	返回值2: 去除[CLS][SEP]且按照常规分词的Attention矩阵
	返回值3: 指定词在 去除[CLS][SEP]且按照常规分词的Attention矩阵 中的行号
	返回值4: tokenization的原始结果(字符串)
	返回值5: tokenization的合并版结果(字符串)
	'''
	# 依赖全局对象bert_tokenizer
	# 依赖全局对象bert_model

	# tokenized句子
	tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
	# 获得tokenized句子的input_id串 '101 2065 3999 2837 ... 102'
	input_ids = tokenized_sent['input_ids'][0]
	# 截掉句子input_id两端的[101] 和 [102]
	input_ids = input_ids[1:-1]
	# 截掉句子两端[CLS]和[SEP]后的tokenization结果
	input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

	# 获得token-word合并映射表
	better_token = complete_tokenizer(sent)
	token_word_id_map_list = better_token['idmap_list']
	token_word_id_map_dict = better_token['idmap_dict']
	
	# 获得目标词在BERT词表中的编号
	target_word_id = bert_tokenizer.convert_tokens_to_ids(word)
	# 获得目标词在token-word合并映射表转换前的attention矩阵中的行号
	target_word_attn_row_num_token =  (input_ids == target_word_id).nonzero()[0]
	# print(f'Target word row No. before dense: {target_word_attn_row_num_token}')
	# 获得目标词在token-word合并映射表转换后的attention矩阵中的行号
	target_word_attn_row_num_dense = token_word_id_map_list[target_word_attn_row_num_token]
	# print(f'Target word row No. after dense: {target_word_attn_row_num_dense}')
	# 获得未经裁边和压缩的attention矩阵
	attn = bert_model(**tokenized_sent,output_attentions=True)['attentions'][attn_layer][0][attn_head].detach()
	# 获得裁边后的attention矩阵
	attn_trimmed = attn_matrix_trim_scale(attn) # 修边attn矩阵
	# 获得裁边并压缩后的attention矩阵(依据token-word合并映射表)
	attn_dense = attn_matrix_denser(token_word_id_map_dict,attn_trimmed) # 修边-合并矩阵
	# 获得目标词在裁边并压缩后attention矩阵中对应的行(tensor)
	# word_attn_dense_row = attn_dense[target_word_attn_row_num_dense].tolist()
	word_attn_dense_row = attn_dense[target_word_attn_row_num_dense]
	# 获得目标词在裁边并压缩后attention矩阵中对应的行(dict: key是位置, value是关注度)
	#word_attn_row_dict = dict(enumerate(word_attn_row.tolist()))
	return (word_attn_dense_row,attn_dense,target_word_attn_row_num_dense)

def get_word_attn_row_num(word,sent):
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

def true_length_sents_selector(word,sent_len,save_path,en_corpus,limit):
	# 筛选BERT分词后,长度等于sent_len的句子
	# 因为BERT_TOKENIZER并不按照空格分词, 在sent_selector中预选长度为n的句子, 在BERT分词后, 长度可能大于n
	# 所以, 从sent_selector中选取长度小于sent_len的句子, 从中会产生BERT分词后, 长度恰好等于sent_len的句子
	# limit限制最多保存多少个句子, 文中取16000
	sents = fix_len_sent_extractor(word,sent_len,en_corpus)
	sents += fix_len_sent_extractor(word,sent_len-1,en_corpus)
	sents += fix_len_sent_extractor(word,sent_len-2,en_corpus)
	sents += fix_len_sent_extractor(word,sent_len-3,en_corpus)
	print(f'No. of sents: {len(sents)}')
	truelen_sents = []
	limit_cnt = 1
	for sent in sents:
		if limit_cnt <=limit:
			# tokenized句子
			res_better_token  = complete_tokenizer(sent)
			true_len  = len(res_better_token['dense_tokens'])
			if true_len == sent_len:
				truelen_sents.append(sent)
				limit_cnt += 1
		else:
			break
	print(f'No. of sents of len {sent_len}: {len(truelen_sents)}')
	savename_truelen_sents = Path(save_path).joinpath(f'truelen_sents_{word}_{sent_len}_{len(truelen_sents)}.pkl')
	with open(savename_truelen_sents,mode='wb') as resfile:
		pickle.dump(truelen_sents,resfile)
	return truelen_sents

# 功能: 主函数, 获取词在句中对应的attn行和位置编号
# 函数ID: 20220403124532 
# 依赖函数: 
def get_word_attn_rows(word,truelen_sents,sent_len,save_path):
	# 根据真正长度等于sent_len的句子, 提取它们中, 各个目标词各自的144行长度为sent_len的attention行
	word_pos = []
	len_truelen_sents = len(truelen_sents)
	# 创建承载容器
	attn_rows = torch.zeros(12,12,len_truelen_sents,sent_len)
	# 获得attention行 和 词在句中位置的标签
	# tsent = truelength_sent
	tsent_cnt = 0
	for tsent in truelen_sents:
		tokenized_sent = bert_tokenizer(tsent,return_tensors='pt')
		res_better_token  = complete_tokenizer(tsent)
		idmap_dict = res_better_token['idmap_dict']
		idmap_list = res_better_token['idmap_list']
		sent_attn_144 = torch.stack(bert_model(**tokenized_sent,output_attentions=True)['attentions']).squeeze()
		word_row_num = get_word_attn_row_num(word,tsent)
		word_pos.append(word_row_num)
		for layer in range(12):
			for head in range(12):
				attn_trimmed = attn_matrix_trim_scale(sent_attn_144[layer][head])
				attn_dense = attn_matrix_denser(idmap_dict,attn_trimmed)
				attn_row = attn_dense[word_row_num].detach()
				attn_rows[layer][head][tsent_cnt]=attn_row
		tsent_cnt += 1
		if tsent_cnt%100 ==0:
				print(f'processed {tsent_cnt} sents')

	# 保存最终结果
	savename_rows = Path(save_path).joinpath(f'attn_rows_{word}_{len_truelen_sents}.pkl')
	savename_labs = Path(save_path).joinpath(f'attn_labs_{word}_{len_truelen_sents}.pkl')
	try:
		with open(savename_rows,mode='wb') as resfilerows:
			pickle.dump(attn_rows,resfilerows)
	except:
		pass

	try:
		with open(savename_labs,mode='wb') as resfilelabs:
			pickle.dump(word_pos,resfilelabs)
	except:
		pass
	return attn_rows,word_pos



# truelensents = true_length_sents_selector(word,length,save_path,en_corpus)
# del en_corpus
# res = get_word_attn_rows(word,truelensents,save_path)