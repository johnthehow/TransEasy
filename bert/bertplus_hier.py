from transformers import BertTokenizer
from transformers import BertModel
from torch.nn import Softmax
import torch
import matplotlib.pyplot as plt

# 设计思想, 高计算量操作尽量减少, 对齐问题解决思路要清晰
# 一次句子分析, 多个函数共享
# 统一用torch.tensor表征所有容器

proxies = {'http':'http://127.0.0.1:1080','https':'https://127.0.0.1:1080'}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class pre_tokenizations:
	def __init__(self, sent):
		self.raw_text = sent.strip() # str
		self.space_tokenization = sent.split(sep=' ')  # [str, str]
		self.raw_text_lower = self.raw_text.lower() # str
		self.space_tokenization_lower = [i.lower() for i in self.space_tokenization] # [str, str]
		# self.obj = tokenizer(self.raw_text_lower, return_tensors='pt') # {key:tensor, key:tensor}
		# self.ids = self.obj['input_ids'][0].tolist() # [int, int]
		# self.wordpieces = tokenizer.convert_ids_to_tokens(self.ids)  # [str,str]
		# self.wordpieces_nosep = self.wordpieces[1:-1] # [str, str]
		# self.ids_nosep = self.ids[1:-1] # [int, int] 
		# self.bert_model_return = model(**self.obj, output_hidden_states=True, output_attentions=True) # (tensor, tensor, tuple, typle) # (last_hidden_state, pooler_output, hidden_states, attentions)
		# self.bert_last_hidden_state = self.bert_model_return[0].squeeze() # tensor
		# self.bert_hidden_states = torch.stack(self.bert_model_return[2][1:],axis=0).squeeze() # tensor
		# self.bert_attentions = torch.stack(self.bert_model_return[3],axis=0).squeeze() # tensor
		# self.nosep = self.bert_attentions[:,:,1:-1,1:-1] # tensor
		# self.nosep_linscale = self.nosep/self.nosep.sum(axis=-1).unsqueeze(-1) # tensor
		# self.nosep_softmaxscale = Softmax(dim=-1)(self.nosep) # tensor
		# self.space_wordpiece_posmap = self.wordpiece_mapping(self.space_tokenization_lower) # {int:[int,int], int:[int,int]}
		# self.reduced_attention = self.reduced_attention_nosep_linscale(self.space_tokenization_lower) # tensor

class bert_tokenizations:
	def __init__(self, sent):
		self.pre_tokenizations = pre_tokenizations(sent)
		self.obj = tokenizer(self.pre_tokenizations.raw_text_lower, return_tensors='pt') # {key:tensor, key:tensor}
		self.ids = self.obj['input_ids'][0].tolist() # [int, int]
		self.wordpieces = tokenizer.convert_ids_to_tokens(self.ids)  # [str,str]
		self.wordpieces_nosep = self.wordpieces[1:-1] # [str, str]
		self.ids_nosep = self.ids[1:-1] # [int, int]

class bert_outputs:
	def __init__(self, sent):
		self.pre_tokens = pre_tokenizations(sent)
		self.bert_tokens = bert_tokenizations(sent)
		self.bert_model_return = model(**self.bert_tokens.obj, output_hidden_states=True, output_attentions=True) # (tensor, tensor, tuple, typle) # (last_hidden_state, pooler_output, hidden_states, attentions)

class hidden_states:
	def __init__(self,sent):
		self.bert_return = bert_outputs(sent).bert_model_return
		self.last = self.bert_return[0].squeeze() # tensor
		self.all = torch.stack(self.bert_return[2][1:],axis=0).squeeze() # tensor

class attentions:
	def __init__(self,sent):
		self.pre_token = tokenizations(sent).pre
		self.pre_token_lower = self.pre_token.space_tokenization_lower
		self.bert_token = tokenizations(sent).bert
		self.bert_token_wordpieces_nosep = self.bert_token.wordpieces_nosep
		self.bert_return = bert_outputs(sent).bert_model_return
		self.raw = torch.stack(self.bert_return[3],axis=0).squeeze() # tensor
		self.nosep = self.raw[:,:,1:-1,1:-1] # tensor
		self.nosep_linscale = self.nosep/self.nosep.sum(axis=-1).unsqueeze(-1) # tensor
		self.nosep_softmaxscale = Softmax(dim=-1)(self.nosep) # tensor
		self.space_wordpiece_posmap = self.wordpiece_mapping(self.pre_token_lower) # {int:[int,int], int:[int,int]}
		self.reduced= self.reduced_attention_nosep_linscale(self.pre_token_lower) # tensor

	def wordpiece_mapping(self,target_tokens):
		'''[int,int]'''
		ud_sent = target_tokens
		wp_sent = self.bert_token_wordpieces_nosep
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

	def reduced_attention_nosep_linscale(self,target_tokens):
		posmap = self.wordpiece_mapping(self.pre_token.space_tokenization_lower)
		num_rows = len(posmap.keys())
		num_cols = sum([len(posmap[x]) for x in posmap if isinstance(posmap[x], list)])
		row_reduced_attention = torch.empty(12,12,num_rows,num_cols)
		row_col_reduced_attention = torch.empty(12,12,num_rows,num_rows)
		for lay in range(12):
			for head in range(12):
				for pair in posmap.items():
					row_reduced_attention[lay][head][pair[0]] = (torch.index_select(self.nosep_linscale[lay][head], -2, torch.tensor(pair[1])).sum(axis=-2))/len(pair[1])
		for lay in range(12):
			for head in range(12):
				for pair in posmap.items():
					row_col_reduced_attention[lay][head][:,pair[0]] = torch.index_select(row_reduced_attention[lay][head], -1, torch.tensor(pair[1])).sum(axis=-1)
		return row_col_reduced_attention
	
class viz:
	def __init__(self,sent):
		self.attns = attentions(sent)
		self.tokens = tokenizations(sent)

	def raw(self,lay,head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.attns.raw[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		xticklabels = [str(i)+' '+j for i,j in zip(range(data.shape[0]), self.tokens.bert.wordpieces)]
		yticklabels = [j+' '+str(i) for i,j in zip(range(data.shape[0]), self.tokens.bert.wordpieces)]
		ax.set_xticklabels(xticklabels,rotation=45, ha='left')
		ax.set_yticklabels(yticklabels)
		ax.set_xlabel(f'Layer-{lay+1}, Head-{head+1}')
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def nosep_linscale(self,lay,head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.attns.nosep_linscale[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		ax.set_xticklabels(self.tokens.bert.wordpieces_nosep, rotation=45)
		ax.set_yticklabels(self.tokens.bert.wordpieces_nosep)
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def nosep_softmaxscale(self,lay,head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.attns.nosep_softmaxscale[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		ax.set_xticklabels(self.tokens.bert.wordpieces_nosep, rotation=45)
		ax.set_yticklabels(self.tokens.bert.wordpieces_nosep)
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def nosep_linscale_reduced(self,lay,head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.attns.reduced[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		ax.set_xticklabels(self.tokens.pre.space_tokenization_lower, rotation=45)
		ax.set_yticklabels(self.tokens.pre.space_tokenization_lower)
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()

class bert:
	def __init__(self,sent):
		self.outputs = bert_outputs(sent)
		self.hidden_states = hidden_states(sent)
		self.attentions = attentions(sent)

class tokenizations:
	def __init__(self,sent):
		self.pre = pre_tokenizations(sent)
		self.bert = bert_tokenizations(sent)

class analyzer:
	def __init__(self,sent):
		self.tokenization = tokenizations(sent)
		self.bert = bert(sent)
		self.viz = viz(sent)

sent = 'The salesman gave us a demo of the vacuum cleaner, and it seemed to work very well.'

analysis = analyzer(sent)

print('done')