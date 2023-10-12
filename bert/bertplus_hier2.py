from transformers import BertTokenizer
from transformers import BertModel
from torch.nn import Softmax
import torch
import matplotlib.pyplot as plt
import os

# 每次Transformer实例化需要联网, 不论是否已经下载模型到本地, 此举在于配置操作系统环境变量
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class space_tokenizations:
	def __init__(self, sent):
		self.raw = sent.strip().lower() # str
		self.spaced = self.raw.split(sep=' ') # [str, str]

class bert_tokenizations:
	def __init__(self, sent):
		self.space_tokenizations = space_tokenizations(sent)
		self.obj = tokenizer(self.space_tokenizations.raw, return_tensors='pt') # {key:tensor, key:tensor}
		self.ids = self.obj['input_ids'][0].tolist() # [int, int]
		self.ids_noclssep = self.ids[1:-1] # [int, int]
		self.wordpieces = bert_wordpieces(sent)

class bert_wordpieces:
	def __init__(self, sent):
		self._pre_tokenizations = space_tokenizations(sent)
		self._obj = tokenizer(self._pre_tokenizations.raw, return_tensors='pt') # {key:tensor, key:tensor}
		self._ids = self._obj['input_ids'][0].tolist() # [int, int]
		self.raw = tokenizer.convert_ids_to_tokens(self._ids)
		self.noclssep = self.raw[1:-1]

class tokenizations: # 20231012142104
	def __init__(self, sent, user_tokenization):
		self.pre = space_tokenizations(sent)
		self.bert = bert_tokenizations(sent)
		self.user = user_tokenization

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

class attentions_noclssep_scale_linear_reduced:
	def __init__(self,  sent, user_tokenization,):
		self.tokenization = tokenizations(sent)
		self.tokenization.custom = user_tokenization


