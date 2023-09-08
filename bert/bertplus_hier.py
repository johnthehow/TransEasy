# 设计思想, 高计算量操作尽量减少, 对齐问题解决思路要清晰
# 一次句子分析, 多个函数共享
# 统一用torch.tensor表征所有容器
# 设计概念图: bertplus_hier_20230906103328.drawio

from transformers import BertTokenizer
from transformers import BertModel
from torch.nn import Softmax
import torch
import matplotlib.pyplot as plt
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class pre_tokenizations:
	def __init__(self, sent):
		self.raw = sent.strip().lower() # str
		self.spaced = self.raw.split(sep=' ') # [str, str]

class bert_tokenizations:
	def __init__(self, sent):
		self.pre_tokenizations = pre_tokenizations(sent)
		self.obj = tokenizer(self.pre_tokenizations.raw, return_tensors='pt') # {key:tensor, key:tensor}
		self.ids = self.obj['input_ids'][0].tolist() # [int, int]
		self.ids_noclssep = self.ids[1:-1] # [int, int]
		self.wordpieces = bert_wordpieces(sent)

class bert_wordpieces:
	def __init__(self, sent):
		self._pre_tokenizations = pre_tokenizations(sent)
		self._obj = tokenizer(self._pre_tokenizations.raw, return_tensors='pt') # {key:tensor, key:tensor}
		self._ids = self._obj['input_ids'][0].tolist() # [int, int]
		self.raw = tokenizer.convert_ids_to_tokens(self._ids)
		self.noclssep = self.raw[1:-1]

class tokenizations:
	def __init__(self,sent):
		self.pre = pre_tokenizations(sent)
		self.bert = bert_tokenizations(sent)
		self.custom = []

	def wordpos(self,word, wordpiece=True, noclssep=True, custom=False):
		if wordpiece == True:
			if noclssep == True:
				return [i for i,j in enumerate(self.bert.wordpieces.raw) if j == word]
			else:
				return [i for i,j in enumerate(self.bert.wordpieces.noclssep) if j == word]
		else:
			if custom == True:
				return [i for i,j in enumerate(self.custom) if j == word]
			else:
				return [i for i,j in enumerate(self.pre.spaced) if j == word]

class bert_outputs:
	def __init__(self, bert_tokenization_obj):
		self.model_return = model(**bert_tokenization_obj, output_hidden_states=True, output_attentions=True) # (tensor, tensor, tuple, typle) # (last_hidden_state, pooler_output, hidden_states, attentions)

class hidden_states:
	def __init__(self,bertout):
		self.bert_return = bertout.model_return
		self.last = self.bert_return[0].squeeze() # tensor
		self.all = torch.stack(self.bert_return[2][1:],axis=0).squeeze() # tensor

class attentions:
	def __init__(self, bert_out, tokens):
		# self.raw = torch.stack(bert_out.model_return[3],axis=0).squeeze() # tensor
		self._tokens = tokens
		self.noclssep = attentions_noclssep(bert_out, tokens)
		self._raw_data = self.noclssep.scale.linear.raw._raw
		self.raw = attentions_raw(self._raw_data, tokens)

class attentions_raw:
	def __init__(self, raw_data, tokens):
		self.matrices = raw_data
		self._tokens = tokens
	
	def viz(self, lay, head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.matrices[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		xticklabels = [str(i+1)+' '+j for i,j in zip(range(data.shape[0]), self._tokens.bert.wordpieces.raw)]
		yticklabels = [j+' '+str(i+1) for i,j in zip(range(data.shape[0]), self._tokens.bert.wordpieces.raw)]
		ax.set_xticklabels(xticklabels, rotation=45, ha='left')
		ax.set_yticklabels(yticklabels)
		ax.set_xlabel(f'Layer-{lay+1}, Head-{head+1}')
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def word_rows(self, word):
		word_poss = self._tokens.wordpos(word, wordpiece=True, noclssep=False)
		rows = torch.index_select(self.matrices, -2, torch.tensor(word_poss))
		return rows

class attentions_noclssep:
	def __init__(self, bert_out, tokens):
		self.scale = attentions_noclssep_scale(bert_out,tokens)
		self.raw = attentions_noclssep_raw(self.scale._noclssep, tokens)

class attentions_noclssep_raw:
	def __init__(self, noclssep_data, tokens):
		self.matrices = noclssep_data
		self._tokens = tokens

	def viz(self, lay, head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.matrices[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		xticklabels = [str(i+1)+' '+j for i,j in zip(range(data.shape[0]), self._tokens.bert.wordpieces.noclssep)]
		yticklabels = [j+' '+str(i+1) for i,j in zip(range(data.shape[0]), self._tokens.bert.wordpieces.noclssep)]
		ax.set_xticklabels(xticklabels, rotation=45, ha='left')
		ax.set_yticklabels(yticklabels)
		ax.set_xlabel(f'Layer-{lay+1}, Head-{head+1}  No [CLS][SEP]')
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def word_rows(self, word):
		word_poss = self._tokens.wordpos(word, wordpiece=True, noclssep=True)
		rows = torch.index_select(self.matrices, -2, torch.tensor(word_poss))
		return rows

class attentions_noclssep_scale:
	def __init__(self, bert_out, tokens):
		self.linear = attentions_noclssep_scale_linear(bert_out, tokens)
		self._noclssep = self.linear.raw._noclssep
		self.softmax = attentions_noclssep_scale_softmax(self._noclssep, tokens)

class attentions_noclssep_scale_softmax:
	def __init__(self, noclssep, tokens):
		self.matrices = Softmax(dim=-1)(noclssep)
		self._tokens = tokens

	def viz(self, lay, head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.matrices[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		xticklabels = [str(i+1)+' '+j for i,j in zip(range(data.shape[0]), self._tokens.bert.wordpieces.noclssep)]
		yticklabels = [j+' '+str(i+1) for i,j in zip(range(data.shape[0]), self._tokens.bert.wordpieces.noclssep)]
		ax.set_xticklabels(xticklabels, rotation=45, ha='left')
		ax.set_yticklabels(yticklabels)
		ax.set_xlabel(f'Layer-{lay+1}, Head-{head+1}  No [CLS][SEP] Softmax-scale')
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def word_rows(self, word):
		word_poss = self._tokens.wordpos(word, wordpiece=True, noclssep=True)
		rows = torch.index_select(self.matrices, -2, torch.tensor(word_poss))
		return rows

class attentions_noclssep_scale_linear:
	def __init__(self, bert_out, tokens):
		self.raw = attentions_noclssep_scale_linear_raw(bert_out, tokens)
		self._raw_matrices = self.raw.matrices
		self.reduced = attentions_noclssep_scale_linear_reduced(self._raw_matrices, tokens) # tensor

class attentions_noclssep_scale_linear_raw:
	def __init__(self,bert_out, tokens):
		self._bert_return = bert_out.model_return
		self._tokens = tokens
		self._raw = torch.stack(self._bert_return[3],axis=0).squeeze() # tensor
		self._noclssep = self._raw[:,:,1:-1,1:-1] # tensor
		self.matrices = self._noclssep/self._noclssep.sum(axis=-1).unsqueeze(-1) # tensor

	def viz(self, lay, head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.matrices[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		xticklabels = [str(i+1)+' '+j for i,j in zip(range(data.shape[0]), self._tokens.bert.wordpieces.noclssep)]
		yticklabels = [j+' '+str(i+1) for i,j in zip(range(data.shape[0]), self._tokens.bert.wordpieces.noclssep)]
		ax.set_xticklabels(xticklabels, rotation=45, ha='left')
		ax.set_yticklabels(yticklabels)
		ax.set_xlabel(f'Layer-{lay+1}, Head-{head+1}  No [CLS][SEP] Linear-scale')
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def word_rows(self, word):
		word_poss = self._tokens.wordpos(word, wordpiece=True, noclssep=True)
		rows = torch.index_select(self.matrices, -2, torch.tensor(word_poss))
		return rows
		
class attentions_noclssep_scale_linear_reduced:
	def __init__(self, attentions_noclssep_scale_linear_raw, tokens):
		self._tokens = tokens
		self._data = attentions_noclssep_scale_linear_raw
		if not self._tokens.custom:
			self._target_tokens = self._tokens.pre.spaced
		else:
			self._target_tokens = self._tokens.custom
		self.matrices = self.reduced_attention_noclssep_linscale(self._target_tokens)

	def wordpiece_mapping(self,target_tokens):
		'''[int,int]'''
		ud_sent = target_tokens
		wp_sent = self._tokens.bert.wordpieces.noclssep
		ud_cnt = 0
		wp_cnt = 0
		wp_buffer = []
		idmap_list = []
		while True:
			try:
				ud_pop = ud_sent[ud_cnt]
				wp_pop = wp_sent[wp_cnt]
				if ud_pop == wp_pop:
					idmap_list.append(ud_cnt)
					ud_cnt += 1
					wp_cnt += 1
				else:
					wp_buffer.append(wp_pop.replace('##',''))
					idmap_list.append(ud_cnt)
					if ''.join(wp_buffer) != ud_pop:
						wp_cnt += 1
					else:
						wp_cnt += 1
						ud_cnt += 1
						wp_buffer = []
			except IndexError:
				break
		idmap_dict = dict()
		for i in range(max(idmap_list)+1):
			idmap_dict[i] = [idx for idx,j in enumerate(idmap_list) if j==i]
		return idmap_dict

	def reduced_attention_noclssep_linscale(self,target_tokens):
		posmap = self.wordpiece_mapping(target_tokens)
		num_rows = len(posmap.keys())
		num_cols = sum([len(posmap[x]) for x in posmap if isinstance(posmap[x], list)])
		row_reduced_attention = torch.empty(12,12,num_rows,num_cols)
		row_col_reduced_attention = torch.empty(12,12,num_rows,num_rows)
		for lay in range(12):
			for head in range(12):
				for pair in posmap.items():
					row_reduced_attention[lay][head][pair[0]] = (torch.index_select(self._data[lay][head], -2, torch.tensor(pair[1])).sum(axis=-2))/len(pair[1])
		for lay in range(12):
			for head in range(12):
				for pair in posmap.items():
					row_col_reduced_attention[lay][head][:,pair[0]] = torch.index_select(row_reduced_attention[lay][head], -1, torch.tensor(pair[1])).sum(axis=-1)
		return row_col_reduced_attention

	def viz(self, lay, head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.matrices[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		xticklabels = [str(i+1)+' '+j for i,j in zip(range(data.shape[0]), self._target_tokens)]
		yticklabels = [j+' '+str(i+1) for i,j in zip(range(data.shape[0]), self._target_tokens)]
		ax.set_xticklabels(xticklabels, rotation=45, ha='left')
		ax.set_yticklabels(yticklabels)
		ax.set_xlabel(f'Layer-{lay+1}, Head-{head+1}  No [CLS][SEP] Linear-scale reduced')
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()		

	def word_rows(self, word):
		word_poss = self._tokens.wordpos(word, wordpiece=False, noclssep=True)
		rows = torch.index_select(self.matrices, -2, torch.tensor(word_poss))
		return rows
	
	@property
	def attention_distance(self): # tensor(12,12) 一句话的144个关注距离
		attn_distances = []
		max_poss = self.matrices.argmax(axis=-1)
		for i in range(self.matrices.shape[3]): # 确定reduced attention matrix的尺寸
			attn_distance = abs(max_poss[:,:,i]-i) # 每行最大值所在位置-行号并取绝对值, 共144个值
			attn_distances.append(attn_distance) # 把每个词的144个关注距离里添加到容器中
		return sum(attn_distances)/self.matrices.shape[3] # 一句话的总依存距离除以句长

	@property
	def standard_attention_distance(self): # tensor(12,12)
		sd_attn_distance = self.attention_distance/self.matrices.shape[3]
		return sd_attn_distance

# 依赖: class:tokenizations
class analyzer:
	def __init__(self,sent, custom_tokenization):
		self.tokenization = tokenizations(sent)
		self.tokenization.custom = custom_tokenization
		self._bertout = bert_outputs(self.tokenization.bert.obj)
		self.hidden_states = hidden_states(self._bertout)
		self.attentions = attentions(self._bertout,self.tokenization)



# sent = 'The salesman gave us a demo of the Huggingface course, and it is a seemmingly working very well.'
# ud_sent = ['the', 'salesman', 'gave', 'us', 'a', 'demo', 'of', 'the', 'huggingface', 'course', ',', 'and', 'it', 'is', 'a', 'seemmingly', 'working', 'very', 'well', '.']
# analysis = analyzer(sent, ud_sent)

# res = analysis.attentions.noclssep.scale.linear.reduced.attention_distance
# print('done')