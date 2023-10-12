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

