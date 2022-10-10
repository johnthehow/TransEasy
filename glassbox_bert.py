import torch
from torch import nn
from transformers import BertModel
from transformers import BertTokenizer

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class glassbox_bert:
    '''输出BERT所有的前向传播阶段表征, 共32项'''
    # 依赖全局对象bert_model (速度考虑, 避免每分析一句话调用一次模型)
    # 依赖全局对象bert_tokenizer (速度考虑, 避免每分析一句话调用一次模型)
    # 仅适用于一句话输入
    # 仅适用于bert-base-uncased
    # 依赖全局对象bert_tokenizer
    # 依赖全局对象bert_model
    def __init__(self,text):
        # 准备阶段/Pre-Embedding阶段/Preemb阶段
        self.model = bert_model
        self.tokenizer = bert_tokenizer
        self.sent = text
        self.__bert_tokenized_text = self.tokenizer(self.sent,return_tensors='pt')

        self.vocab_emb = self.model.embeddings.word_embeddings.weight
        self.token_ids = self.__bert_tokenized_text['input_ids'][0]
        self.tokens = self.tokenizer.convert_ids_to_tokens(self.token_ids.tolist())
        self.seq_len = len(self.token_ids)

        self.pipeline_res = self.model(**self.__bert_tokenized_text,output_hidden_states=True,output_attentions=True)
        self.pipeline_attns = self.pipeline_res['attentions']
        self.pipeline_hiddens = self.pipeline_res['hidden_states']

        self.preemb_word_emb = torch.zeros(self.seq_len,768)
        cnt = 0
        for id in self.token_ids:
            self.preemb_word_emb[cnt] = self.vocab_emb[id]
            cnt +=1
        del cnt
        self.preemb_pos_emb = self.model.embeddings.position_embeddings.weight[:self.seq_len]
        self.preemb_seg_emb = self.model.embeddings.token_type_embeddings.weight[0].repeat(self.seq_len,1)
        self.preemb_sum_emb = self.preemb_word_emb + self.preemb_pos_emb + self.preemb_seg_emb 
        self.preemb_norm_sum_emb = self.model.embeddings.LayerNorm(self.preemb_sum_emb)

        self.preemb_hidden = self.preemb_norm_sum_emb # n×768

        # 多层结果承载变量
        self.preencoder_hidden = torch.zeros(13,self.seq_len,768)
        self.preencoder_hidden[0] = self.preemb_hidden


        # Multi-Head Self-Attention阶段/SelfAttn阶段
        self.selfattn_query_weight = torch.zeros(12,12,64,768)
        self.selfattn_key_weight = torch.zeros(12,12,64,768)
        self.selfattn_value_weight = torch.zeros(12,12,64,768)

        self.selfattn_query_bias = torch.zeros(12,12,64)
        self.selfattn_key_bias = torch.zeros(12,12,64)
        self.selfattn_value_bias = torch.zeros(12,12,64)

        self.selfattn_query_hidden = torch.zeros(12,12,self.seq_len,64)
        self.selfattn_key_hidden = torch.zeros(12,12,self.seq_len,64)
        self.selfattn_value_hidden = torch.zeros(12,12,self.seq_len,64)

        self.selfattn_qkt_hidden = torch.zeros(12,12,self.seq_len,self.seq_len)
        self.selfattn_qkt_scale_hidden = torch.zeros(12,12,self.seq_len,self.seq_len)
        self.selfattn_qkt_scale_soft_hidden = torch.zeros(12,12,self.seq_len,self.seq_len)
        self.selfattn_attention = self.selfattn_qkt_scale_soft_hidden
        self.selfattn_contvec = torch.zeros(12,12,self.seq_len,64)
        self.selfattn_contvec_concat = torch.zeros(12,self.seq_len,768)

        # 方案1(手动乘积: 正确)
        self.addnorm1_dense_weight = torch.zeros(12,768,768)
        self.addnorm1_dense_bias = torch.zeros(12,768)
        self.addnorm1_dense_hidden = torch.zeros(12,self.seq_len,768)
        self.addnorm1_add_hidden = torch.zeros(12,self.seq_len,768)
        self.addnorm1_norm_hidden = torch.zeros(12,self.seq_len,768)

        self.ffnn_dense_weight = torch.zeros(12,768,3072)
        self.ffnn_dense_bias = torch.zeros(12,3072)
        self.ffnn_dense_hidden = torch.zeros(12,self.seq_len,3072)
        self.ffnn_dense_act = torch.zeros(12,self.seq_len,3072)

        self.addnorm2_dense_weight = torch.zeros(12,3072,768)
        self.addnorm2_dense_bias = torch.zeros(12,768)
        self.addnorm2_dense_hidden = torch.zeros(12,self.seq_len,768)
        self.addnorm2_add_hidden = torch.zeros(12,self.seq_len,768)
        self.addnorm2_norm_hidden = torch.zeros(12,self.seq_len,768)

        for lay in range(12):
            for head in range(12):
            # Multi-Head Self-attention阶段/selfattn阶段
                self.selfattn_query_weight[lay][head] = self.model.encoder.layer[lay].attention.self.query.weight[head*64:(head+1)*64,:] # 64*768
                self.selfattn_key_weight[lay][head] = self.model.encoder.layer[lay].attention.self.key.weight[head*64:(head+1)*64,:] # 64*768
                self.selfattn_value_weight[lay][head] = self.model.encoder.layer[lay].attention.self.value.weight[head*64:(head+1)*64,:] # 64*768

                self.selfattn_query_bias[lay][head] = self.model.encoder.layer[lay].attention.self.query.bias[head*64:(head+1)*64] # 64
                self.selfattn_key_bias[lay][head] = self.model.encoder.layer[lay].attention.self.key.bias[head*64:(head+1)*64] # 64
                self.selfattn_value_bias[lay][head] = self.model.encoder.layer[lay].attention.self.value.bias[head*64:(head+1)*64] # 64

                self.selfattn_query_hidden[lay][head] = self.preencoder_hidden[lay]@self.selfattn_query_weight[lay][head].T+self.selfattn_query_bias[lay][head] # XW.T+b = n*768*768*64 = n*64
                self.selfattn_key_hidden[lay][head] = self.preencoder_hidden[lay]@self.selfattn_key_weight[lay][head].T+self.selfattn_key_bias[lay][head] # XW.T+b = n*768*768*64 = n*64
                self.selfattn_value_hidden[lay][head] = self.preencoder_hidden[lay]@self.selfattn_value_weight[lay][head].T+self.selfattn_value_bias[lay][head] # XW.T+b = n*768*768*64 = n*64

                self.selfattn_qkt_hidden[lay][head] = self.selfattn_query_hidden[lay][head]@self.selfattn_key_hidden[lay][head].T # qk.t = n*64*64*n = n*n 
                self.selfattn_qkt_scale_hidden[lay][head] = self.selfattn_qkt_hidden[lay][head]/8 # n*n
                self.selfattn_qkt_scale_soft_hidden[lay][head] = nn.Softmax(dim=-1)(self.selfattn_qkt_scale_hidden[lay][head]) # n*n
                self.selfattn_contvec[lay][head] = self.selfattn_qkt_scale_soft_hidden[lay][head]@self.selfattn_value_hidden[lay][head] # attn*value_hidden = n*n*n*64 = n*64
            
            self.selfattn_contvec_concat[lay] = torch.cat([block for block in self.selfattn_contvec[lay]],axis=-1) # 12*n*768

            # Add & Norm 1阶段/addnorm1阶段
            self.addnorm1_dense_weight[lay] = self.model.encoder.layer[lay].attention.output.dense.weight.T # 768*768 # 关键步骤, weight一定要转置
            self.addnorm1_dense_bias[lay] = self.model.encoder.layer[lay].attention.output.dense.bias # 768
            self.addnorm1_dense_hidden[lay] = self.selfattn_contvec_concat[lay]@self.addnorm1_dense_weight[lay]+self.addnorm1_dense_bias[lay] # n*768*768*768 = n*768
            self.addnorm1_add_hidden[lay] = self.addnorm1_dense_hidden[lay] + self.preencoder_hidden[lay] # n*768
            self.addnorm1_norm_hidden[lay] = self.model.encoder.layer[lay].attention.output.LayerNorm(self.addnorm1_add_hidden[lay]) # n*768
            # Feedforward阶段/ffnn阶段
            self.ffnn_dense_weight[lay] = self.model.encoder.layer[lay].intermediate.dense.weight.T # 768*3072 # 关键步骤, weight一定要转置
            self.ffnn_dense_bias[lay] = self.model.encoder.layer[lay].intermediate.dense.bias # 3072
            self.ffnn_dense_hidden[lay] = self.addnorm1_norm_hidden[lay]@self.ffnn_dense_weight[lay]+self.ffnn_dense_bias[lay] # n*768*768*3072 = n*3072
            self.ffnn_dense_act[lay] = self.model.encoder.layer[lay].intermediate.intermediate_act_fn(self.ffnn_dense_hidden[lay]) # n*3072
            # Add & Norm 2阶段/addnorm2阶段
            self.addnorm2_dense_weight[lay] = self.model.encoder.layer[lay].output.dense.weight.T # 3072*768 # 关键步骤, weight一定要转置
            self.addnorm2_dense_bias[lay] = self.model.encoder.layer[lay].output.dense.bias # 768
            self.addnorm2_dense_hidden[lay] = self.ffnn_dense_act[lay]@self.addnorm2_dense_weight[lay]+self.addnorm2_dense_bias[lay] # n*768
            self.addnorm2_add_hidden[lay] = self.addnorm2_dense_hidden[lay] + self.addnorm1_norm_hidden[lay] # n*768
            self.addnorm2_norm_hidden[lay] = self.model.encoder.layer[lay].output.LayerNorm(self.addnorm2_add_hidden[lay]) # n*768
            self.preencoder_hidden[lay+1] = self.addnorm2_norm_hidden[lay]