from transformers import BertTokenizer
from transformers import BertModel
from torch.nn import Softmax
import torch
import matplotlib.pyplot as plt

# 设计思想, 高计算量操作尽量减少, 对齐问题解决思路要清晰
# 一次句子分析, 多个函数共享
# 统一用torch.tensor表征所有容器


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class sent_analyzer:
	def __init__(self, sent):
		self.raw_text = sent.strip() # str
		self.space_tokenization = sent.split(sep=' ')  # [str, str]
		self.raw_text_lower = self.raw_text.lower() # str
		self.space_tokenization_lower = [i.lower() for i in self.space_tokenization] # [str, str]
		self.bert_tokenization_obj = tokenizer(self.raw_text_lower, return_tensors='pt') # {key:tensor, key:tensor}
		self.bert_tokenization_ids = self.bert_tokenization_obj['input_ids'][0].tolist() # [int, int]
		self.bert_tokenization_wordpieces = tokenizer.convert_ids_to_tokens(self.bert_tokenization_ids)  # [str,str]
		self.bert_tokenization_wordpieces_nosep = self.bert_tokenization_wordpieces[1:-1] # [str, str]
		self.bert_tokenization_ids_nosep = self.bert_tokenization_ids[1:-1] # [int, int] 
		self.bert_model_return = model(**self.bert_tokenization_obj, output_hidden_states=True, output_attentions=True) # (tensor, tensor, tuple, typle) # (last_hidden_state, pooler_output, hidden_states, attentions)
		self.bert_last_hidden_state = self.bert_model_return[0].squeeze() # tensor
		self.bert_hidden_states = torch.stack(self.bert_model_return[2][1:],axis=0).squeeze() # tensor
		self.bert_attentions = torch.stack(self.bert_model_return[3],axis=0).squeeze() # tensor
		self.bert_attentions_nosep = self.bert_attentions[:,:,1:-1,1:-1] # tensor
		self.bert_attentions_nosep_linscale = self.bert_attentions_nosep/self.bert_attentions_nosep.sum(axis=-1).unsqueeze(-1) # tensor
		self.bert_attentions_nosep_softmaxscale = Softmax(dim=-1)(self.bert_attentions_nosep) # tensor
		self.space_wordpiece_posmap = self.wordpiece_mapping(self.space_tokenization_lower) # {int:[int,int], int:[int,int]}
		self.reduced_attention = self.reduced_attention_nosep_linscale(self.space_tokenization_lower) # tensor
	
	def wordpiece_mapping(self,target_tokens):
		'''[int,int]'''
		ud_sent = target_tokens
		wp_sent = self.bert_tokenization_wordpieces_nosep
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
		posmap = self.wordpiece_mapping(self.space_tokenization_lower)
		num_rows = len(posmap.keys())
		num_cols = sum([len(posmap[x]) for x in posmap if isinstance(posmap[x], list)])
		row_reduced_attention = torch.empty(12,12,num_rows,num_cols)
		row_col_reduced_attention = torch.empty(12,12,num_rows,num_rows)
		for lay in range(12):
			for head in range(12):
				for pair in posmap.items():
					row_reduced_attention[lay][head][pair[0]] = (torch.index_select(self.bert_attentions_nosep_linscale[lay][head], -2, torch.tensor(pair[1])).sum(axis=-2))/len(pair[1])
		for lay in range(12):
			for head in range(12):
				for pair in posmap.items():
					row_col_reduced_attention[lay][head][:,pair[0]] = torch.index_select(row_reduced_attention[lay][head], -1, torch.tensor(pair[1])).sum(axis=-1)
		return row_col_reduced_attention
	
	def viz_bert_attentions(self,lay,head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.bert_attentions[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		xticklabels = [str(i)+' '+j for i,j in zip(range(data.shape[0]), self.bert_tokenization_wordpieces)]
		yticklabels = [j+' '+str(i) for i,j in zip(range(data.shape[0]), self.bert_tokenization_wordpieces)]
		ax.set_xticklabels(xticklabels,rotation=45, ha='left')
		ax.set_yticklabels(yticklabels)
		ax.set_xlabel(f'Layer-{lay+1}, Head-{head+1}')
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def viz_bert_attentions_nosep_linscale(self,lay,head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.bert_attentions_nosep_linscale[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		ax.set_xticklabels([str(i)+' '+j for i,j in zip(range(data.shape[0]), self.bert_tokenization_wordpieces_nosep)], rotation=45)
		ax.set_yticklabels([j+' '+str(i) for i,j in zip(range(data.shape[0]), self.bert_tokenization_wordpieces_nosep)])
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def viz_bert_attentions_nosep_softmaxscale(self,lay,head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.bert_attentions_nosep_softmaxscale[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		ax.set_xticklabels([str(i)+' '+j for i,j in zip(range(data.shape[0]), self.bert_tokenization_wordpieces_nosep)], rotation=45)
		ax.set_yticklabels([j+' '+str(i) for i,j in zip(range(data.shape[0]), self.bert_tokenization_wordpieces_nosep)])
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()
		return

	def viz_bert_attentions_nosep_linscale_reduced(self,lay,head):
		fig = plt.figure()
		ax = fig.subplots()
		data = self.reduced_attention[lay][head].detach().numpy()
		gcf = ax.matshow(data,cmap='Greens')
		ax.set_xticks(range(data.shape[0]))
		ax.set_yticks(range(data.shape[1]))
		ax.set_xticklabels([str(i)+' '+j for i,j in zip(range(data.shape[0]), self.space_tokenization)], rotation=45)
		ax.set_yticklabels([j+' '+str(i) for i,j in zip(range(data.shape[0]), self.space_tokenization)])
		bar = plt.gcf().colorbar(gcf)
		plt.show()
		plt.close()


sent = 'mary had a little lamb, its fleece was white as snow'

sent_obj = sent_analyzer(sent)

print('done')