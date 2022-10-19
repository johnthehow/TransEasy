# 本脚本20221013162019
# attention是一行对应一个单词, 因为每行相加等于1, 列则不然
# tensor.tolist()会产生误差
# >>> bertanomy.bert_tokenizer.convert_tokens_to_string(['hugging','##face'])
# 'huggingface'
# 先BertTokenzer.convert_tokens_to_ids(sent), 再BertTokenzer.convert_ids_to_tokens(tokenids)并不一定能得到原来的句子, 1. 有的词可能被拆分 2.表外词转换回来统统为 [UKN]

import torch
import numpy
import random
import matplotlib.pyplot as plt
from torch import nn
from transformers import BertModel
from transformers import BertTokenizer
from transformers import logging
from sklearn.manifold import TSNE

logging.set_verbosity_error()

bert_model = BertModel.from_pretrained('bert-base-uncased') # 20221013172250
bert_model.eval()
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 20221013172247

# sym_组
def sym_better_tokenizer(sent, trim=True, verbose=False): # 20221013142808
    '''BertTokenizer的好用和全面版pipeline'''
    # [依赖]
    # [返回值]
      # 4组9项的字典
            # 1 第一组1: tokenized_sent: 原生bert_tokenizer返回的对象
            # 2 第二组1: notrim_nomerge_token_ids 原生bert_tokenizer返回的 token ids
            # 3 第二组2: trim_nomerge_token_ids 原生bert_tokenizer返回的 token ids, 除去[CLS]和[SEP]对应的ID
            # 4 第三组1: notrim_nomerge_tokens 原生bert_tokenizer返回的 tokens
            # 5 第三组2: notrim_merge_tokens 原生bert_tokenizer返回的 tokens, 将wordpiece合并
            # 6 第三组3: trim_nomerge_tokens 原生bert_tokenizer返回的 tokens, 除去[CLS]和[SEP]
            # 7 第三组4: trim_merge_tokens 原生bert_tokenizer返回的 tokens, 除去[CLS]和[SEP], 并将wordpiece合并
            # 8 第四组1: wordno_to_pieceno_list 从词到wordpiece的merge映射表,list版, 内容取决于trim开关
            # 9 第四组2: wordno_to_pieceno_dict 从词到wordpiece的merge映射表,list版, 内容取决于trim开关
    def wordno_to_pieceno(trimmed_inputids): # 20221013142816
        cnt = -1
        wordno_to_pieceno_list = [] 
        for tokenid in trimmed_inputids: # token-word合并映射表 list
            if bert_tokenizer.convert_ids_to_tokens([tokenid])[0].startswith('##'):
                cnt = cnt
            else:
                cnt += 1
            wordno_to_pieceno_list.append(cnt)
        # 获得token-word合并映射表:dict版
        wordno_to_pieceno_dict = dict()
        for wordid in wordno_to_pieceno_list:
            wordno_to_pieceno_dict[wordid] = [tokenid for tokenid,realwordid in enumerate(wordno_to_pieceno_list) if realwordid==wordid]
        return {'wordno_to_pieceno_list': wordno_to_pieceno_list, 'wordno_to_pieceno_dict': wordno_to_pieceno_dict}

    
    def wordpiece_adhesive(tokenlist,maplist): # 20221013142819
        mapdict = dict()
        for i in maplist:
            mapdict[i] = [k for k,v in enumerate(maplist) if v==i]
        tokendict = dict()
        for k in mapdict:
            tokendict[k] = [tokenlist[token] for token in mapdict[k]]
        tokendictj = dict()
        for item in tokendict.items():
            notrim_nomerge_tokens = item[1]
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
    notrim_nomerge_token_ids = tokenized_sent['input_ids'][0]
    # 截掉句子input_id两端的[101] 和 [102]
    trim_nomerge_token_ids = notrim_nomerge_token_ids[1:-1]
    # wordpiece合并映射表
    idmap = wordno_to_pieceno(trim_nomerge_token_ids)
    wordno_to_pieceno_list = idmap['wordno_to_pieceno_list']
    wordno_to_pieceno_dict = idmap['wordno_to_pieceno_dict']
    # 截掉句子两端[CLS]和[SEP]后的tokenization结果
    raw_input_tokens = bert_tokenizer.convert_ids_to_tokens(notrim_nomerge_token_ids)
    notrim_merge_tokens = bert_tokenizer.convert_tokens_to_string(raw_input_tokens).split(sep=' ')
    trimmed_input_tokens = raw_input_tokens[1:-1]
    dense_input_tokens_dict = wordpiece_adhesive(trimmed_input_tokens,wordno_to_pieceno_list)
    dense_input_tokens_list = list(dense_input_tokens_dict.values())
    
    if trim == True:
        wordno_to_pieceno_list = wordno_to_pieceno_list
        wordno_to_pieceno_dict = wordno_to_pieceno_dict
    if trim == False:
        wordno_to_pieceno_list = [i+1 for i in wordno_to_pieceno_list]
        wordno_to_pieceno_list = [0] + wordno_to_pieceno_list
        wordno_to_pieceno_list = wordno_to_pieceno_list + [wordno_to_pieceno_list[-1]+1]
        wordno_to_pieceno_dict = dict()
        wordno_to_pieceno_dict_pre = [[idx for idx,value in enumerate(wordno_to_pieceno_list) if value==key] for key in set(wordno_to_pieceno_list)]
        for i in set(wordno_to_pieceno_list):
            wordno_to_pieceno_dict[i] = wordno_to_pieceno_dict_pre[i]


    res = {'tokenized_sent':tokenized_sent,'notrim_nomerge_token_ids':notrim_nomerge_token_ids,'trim_nomerge_token_ids':trim_nomerge_token_ids,'notrim_nomerge_tokens':raw_input_tokens,'notrim_merge_tokens':notrim_merge_tokens, 'trim_nomerge_tokens':trimmed_input_tokens,'trim_merge_tokens': dense_input_tokens_list,'wordno_to_pieceno_list':wordno_to_pieceno_list,'wordno_to_pieceno_dict':wordno_to_pieceno_dict}
    if verbose == True:
        print(f'\n[Message from sym_better_tokenizer]')

        print(f'notrim_nomerge_token_ids: {str(res["notrim_nomerge_token_ids"])}')
        print(f'trim_nomerge_token_ids: {str(res["trim_nomerge_token_ids"])}')

        print(f'notrim_nomerge_tokens: {str(res["notrim_nomerge_tokens"])}')
        print(f'notrim_merge_tokens: {notrim_merge_tokens}')
        print(f'trim_nomerge_tokens: {str(res["trim_nomerge_tokens"])}')
        print(f'trim_merge_tokens: {str(res["trim_merge_tokens"])}')

        print(f'wordno_to_pieceno_list: {str(res["wordno_to_pieceno_list"])}')
        print(f'wordno_to_pieceno_dict: {str(res["wordno_to_pieceno_dict"])}')
    return res

def sym_bert_length_sents_selector(word,sent_len,sent_max,n_shorter,corpus_path): # 20221013142808
    '''返回句子中, bert分词后为指定长度的句子'''
    # [解释]
        # 在不考虑[CLS]和[SEP]的情况下, 一个句子有三种句长
            # 1. 简单空格分词句长
            # 2. wordpiece分词后的句长
            # 3. wordpiece分词后再合并的句长
        # 取第三种句长作为实际句长筛选的标准
        # 因为BERT_TOKENIZER并不按照空格分词, 在sent_selector中预选长度为n的句子, 在BERT分词后, 长度可能大于n
        # 所以, 从sent_selector中选取长度小于sent_len的句子, 从中会产生BERT分词后, 长度恰好等于sent_len的句子
    # [输入]
        # word: 句子中必须含有什么单词
        # sent_len: 目标的句长(BERT标准)是多少
        # sent_max: 最多返回多少个句子
        # n_shorter: 向下取比目标句长短多少个词的句子(空格标准)作为句子来源
        # corpus_path: 一行一句型语料库的位置
    # [依赖]
        # 依赖better_tokenizer 20221013142808
    def space_length_sents_selector(word,sent_len,corpus_path):
        with open(corpus_path, mode='r',encoding='utf-8') as corpus_file:
            corpus_lines = corpus_file.readlines()
            corpus_lines = [line.strip('\n').lower() for line in corpus_lines]
            space_tokenized_corpus_lines = [line.split(sep=' ') for line in corpus_lines]
            word_space_tokenized_corpus_lines = [tkline for tkline in space_tokenized_corpus_lines if word in tkline]
            len_word_space_tokenized_corpus_lines = [tkline for tkline in word_space_tokenized_corpus_lines if len(tkline) == sent_len]
            result_lines = [' '.join(tkline) for tkline in len_word_space_tokenized_corpus_lines]
        return result_lines
    def short_length_sents_selector(word,sent_len,n_shorter,corpus_path):
        multi_len_sents = []
        for i in range(n_shorter+1):
            multi_len_sents += space_length_sents_selector(word, sent_len-i, corpus_path)
        return multi_len_sents
    def bert_length_sents_selector(sent_len,sents,sent_max):
        bert_len_sents = []
        sent_cnt = 0
        for sent in sents:
            if len(sym_better_tokenizer(sent)['trim_merge_tokens']) == sent_len and len(bert_len_sents) <= sent_max:
                bert_len_sents.append(sent)
                sent_cnt += 1
        return bert_len_sents

    multi_len_sents = short_length_sents_selector(word,sent_len,n_shorter,corpus_path)
    bert_len_sents = bert_length_sents_selector(sent_len,multi_len_sents,sent_max)
    return bert_len_sents

# preemb_组
def preemb_word_preemb(word): # 20221013142717
    '''获得一个词的bert预训练静态词向量'''
    # 依赖全局对象bert_model_20221013172250
    # 依赖全局对象bert_tokenizer_20221013172247
    
    word_idx = bert_tokenizer.convert_tokens_to_ids(word)
    if word_idx == 100:
        print('out-of-vocabulary word')
    we = bert_model.embeddings.word_embeddings.weight
    word_preemb = we[word_idx]
    return word_preemb

def preemb_similar_preemb(tensor_vec,tensor_target_vecs,topn):
    ''' 寻找一个BERT词向量和BERT预训练词向量中最相似的那个'''
    # 20221013142743
    # 参数tensor_vec: 已知词向量
    # 参数tensor_target_vecs: BERT预训练词向量们
    # 参数topn: 返回前几个
    import scipy
    query = tensor_vec.detach().numpy()
    keys = tensor_target_vecs.detach().numpy()
    cos = []
    for key in keys:
        cos.append(1-scipy.spatial.distance.cosine(query,key))
    topncos = numpy.argsort(cos)[-topn:].tolist()
    for i in topncos:
        topcos_word = bert_tokenizer.convert_ids_to_tokens(i)
        print(topcos_word)
    return

# attn_组
def attn_sent_attention_matrix(sent,trim_scale=False,merge=False): #20221018103823
    '''获得一句话的attention矩阵'''
    # [输入]
        # trim_scale开关: 是否削去attention两端[SEP]和[CLS]对应的行列, 并将每行线性放缩(每个维度值除以行总和)
        # merge开关: 是否合并wordpiece型词对应的行和列, 行合并后, 新行的值是两行的均值, 新列的值是两列的和(为了保证每行之和仍为1)
    # [输出]
        # (tensor) 一句话的attention矩阵
    # [依赖]
        # bert_tokenizer
        # bert_model
        # sym_better_tokenizer
    # [被依赖]
    # [备注]
        # 代码设计图 NOTION: 20221018160815
    tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
    pipe_attn = torch.squeeze(torch.stack(bert_model(**tokenized_sent,output_attentions=True)['attentions'],dim=0))
    if trim_scale == True:
        if merge == True:
            trim_attn = pipe_attn[:,:,1:-1,1:-1]
            trim_scale_attn = torch.zeros(trim_attn.shape)
            for layer in range(12):
                for head in range(12):
                    for row in range(trim_attn.shape[2]):
                        trim_scale_attn[layer,head,row] = trim_attn[layer][head][row]/(trim_attn[layer][head][row].sum())
            wordno_to_pieceno_dict = sym_better_tokenizer(sent,trim=True)['wordno_to_pieceno_dict']
            trim_scale_rowmerge_attn = torch.zeros(12,12,len(wordno_to_pieceno_dict.keys()),trim_scale_attn.shape[3])
            for layer in range(12):
                for head in range(12):
                    for row in wordno_to_pieceno_dict:
                        trim_scale_rowmerge_attn[layer][head][row] = trim_scale_attn[layer][head][wordno_to_pieceno_dict[row]].sum(axis=0)/len(wordno_to_pieceno_dict[row])

            trim_scale_rowmerge_colmerge_attn = torch.zeros(12,12,len(wordno_to_pieceno_dict.keys()),len(wordno_to_pieceno_dict.keys()))
            for layer in range(12):
                for head in range(12):
                    for col in wordno_to_pieceno_dict:
                        trim_scale_rowmerge_colmerge_attn[layer][head][:,col] = trim_scale_rowmerge_attn[layer][head][:,wordno_to_pieceno_dict[col]].sum(axis=1)
            res = trim_scale_rowmerge_colmerge_attn
        else:
            trim_attn = pipe_attn[:,:,1:-1,1:-1]
            trim_scale_attn = torch.zeros(trim_attn.shape)
            for layer in range(12):
                for head in range(12):
                    for row in range(trim_attn.shape[2]):
                        trim_scale_attn[layer,head,row] = trim_attn[layer][head][row]/(trim_attn[layer][head][row].sum())
            res = trim_scale_attn
    else:
        if merge == True:
            wordno_to_pieceno_dict = sym_better_tokenizer(sent,trim=False)['wordno_to_pieceno_dict']
            notrimscale_rowmerge_attn = torch.zeros(12,12,len(wordno_to_pieceno_dict.keys()),pipe_attn.shape[3])
            for layer in range(12):
                for head in range(12):
                    for row in wordno_to_pieceno_dict:
                        notrimscale_rowmerge_attn[layer][head][row] = pipe_attn[layer][head][wordno_to_pieceno_dict[row]].sum(axis=0)/len(wordno_to_pieceno_dict[row])
            notrimscale_rowmerge_colmerge_attn = torch.zeros(12,12,len(wordno_to_pieceno_dict.keys()),len(wordno_to_pieceno_dict.keys()))
            for layer in range(12):
                for head in range(12):
                    for col in wordno_to_pieceno_dict:
                        notrimscale_rowmerge_colmerge_attn[layer][head][:,col] = notrimscale_rowmerge_attn[layer][head][:,wordno_to_pieceno_dict[col]].sum(axis=1)
            res = notrimscale_rowmerge_colmerge_attn
        else:
            res = pipe_attn

    return res

def attn_word_attention_row(word,sent,trim=True,merge=True,first=True): # 20221013142830
    ''' 返回 指定词 在指定句子中 对应的 attention行'''
    # [输入]
        # trim开关: 是否在除去[CLS]和[SEP]的attention矩阵中查找行
        # merge开关: 在含wordpiece组合型词的句子中, 是否合并wordpiece
        # first开关: 在含wordpiece组合型词的句子中, wordpiece组合词的位置取其第一个位置还是所有位置
    # [输出]
        # (tensor) 12×12×句长: 一个词在144个head中对应的所有attention行
    # [依赖]
        # attn_sent_attention_matrix
        # prop_word_position_in_senet
    # [被依赖]
        # attndistance(word,sent,attn_layer,attn_head): # 20221013
        # attnpos_batch(word:str,sents:list): # 20221013142910
    assert len(word.split(sep=' ')) == 1, print('\n[attn_word_attention_row]\nmultiword phrase not allowed')

    attn_matrices = attn_sent_attention_matrix(sent,trim_scale=trim,merge=merge)
    word_row_no = prop_word_position_in_senet(word,sent,trim=trim,merge=merge,first=first)
    res = attn_matrices[:,:,word_row_no]
    return res    

# hidden_组
def hidden_sent_hidden_states(sent,merge=False): # 20221017151232
    '''获得一句话的hidden states矩阵(13层)\nmerge开关: 是否将wordpiece组合的hidden states合并为一个\ntrim开关: 是否削去[CLS][SEP]对应的hidden states'''
    # [依赖]
        # pipe_pipeline 20221013142729
        # sym_better_tokenizer 20221013142808
    if merge == False:
        res = pipe_pipeline(sent)['hidden_states']
    if merge == True:
        hidden_states = pipe_pipeline(sent)['hidden_states']
        wordno_to_pieceno_dict = sym_better_tokenizer(sent,trim=False)['wordno_to_pieceno_dict']
        new_tensor = torch.zeros(13,len(wordno_to_pieceno_dict.keys()),768) # 一个承载tensor
        for layer in range(13):
            for row in range(len(wordno_to_pieceno_dict.keys())):
                new_tensor[layer][row] = hidden_states[layer][wordno_to_pieceno_dict[row]].sum(axis=0)/len(wordno_to_pieceno_dict[row])
        res = new_tensor
    return res

def hidden_word_hidden_states_in_sent(word,sent,merge=False,first=False): # 20221013165325
    ''' 返回一句话中 一个词 在13个层的hidden_states, 如果有多次出现, 就有多个返回'''
    # [输入]
        # merge开关: wordpiece组合型词是否合并
        # first开关: wordpiece组合型词取第一个词作为代表, 还是合并为一个hidden state
    # [依赖]
        # 依赖全局对象 bert_tokenizer
        # 依赖函数 hidden_sent_hidden_states
        # 依赖函数 prop_word_position_in_senet
    # [返回]
        # 数据类型 tensor
        # 尺寸 13×n×768
    # [备注]
        # 代码设计图 NOTION: 20221018160815
    
    assert len(word.split(sep=' ')) == 1, print('more than one word not allowed')
    is_simplex = len(bert_tokenizer(word)['input_ids']) == 3 # 是否为单纯词(即不被拆成word-piece的词)
    is_oov = bert_tokenizer.convert_tokens_to_ids(word) == 100

    if is_simplex:
        if is_oov:
            res = None
            print('\n[hidden_word_hidden_states_in_sent]\noov token not allowed')
        else:
            if merge == True:
                sent_hidden_states = hidden_sent_hidden_states(sent,merge=True)
                word_pos = prop_word_position_in_senet(word,sent,trim=False,merge=True,first=True)
                res = sent_hidden_states[:,word_pos,:]
            else:
                sent_hidden_states = hidden_sent_hidden_states(sent,merge=False)
                word_pos = prop_word_position_in_senet(word,sent,trim=False,merge=False,first=True)
                res = sent_hidden_states[:,word_pos,:]
    else:
        if merge == True:
            if first == True:
                sent_hidden_states = hidden_sent_hidden_states(sent,merge=True)
                word_pos = prop_word_position_in_senet(word,sent,trim=False,merge=True,first=True)
                res = sent_hidden_states[:,word_pos,:]
            else:
                sent_hidden_states = hidden_sent_hidden_states(sent,merge=True)
                word_pos = prop_word_position_in_senet(word,sent,trim=False,merge=True,first=False)
                res = sent_hidden_states[:,word_pos,:]
        else:
            if first == True:
                sent_hidden_states = hidden_sent_hidden_states(sent,merge=False)
                word_pos = prop_word_position_in_senet(word,sent,trim=False,merge=False,first=True)
                res = sent_hidden_states[:,word_pos,:]
            else:
                sent_hidden_states = hidden_sent_hidden_states(sent,merge=False)
                word_pos = prop_word_position_in_senet(word,sent,trim=False,merge=False,first=False)
                res = sent_hidden_states[:,word_pos,:]

    return res

def hidden_sent_hidden_vector(sent): # 20221014105633
    '''获得一句话的句向量, 使用平均池化法'''
    sent_mat = pipe_pipeline(sent)['last_hidden_state']
    sent_sum = sent_mat.sum(axis=0)
    sent_avg = sent_sum/(sent_mat.shape[0])
    return sent_avg

# stat_组
def prop_word_attention_distance(word,sent,first=False,absolute=True): # 20221013142901
    '''20221018163854 获得一个词在一句话中的关注距离, 只考虑trim_scale和merge后的情况'''
    # [输入]
        # first开关: 是否值考虑一个词的第一个出现
        # absolute开关: 计算绝对关注距离还是相对关注距离(除以句长)
    # [输出]
        # (tensor) 一个词在一句话中 在所有head中的关注距离
    # [依赖]
        # attn_word_attention_row
        # sym_better_tokenizer
        # prop_word_position_in_senet
    assert len(word.split(sep=' ')) == 1, print('\n[prop_word_attention_distance]\nmore than one word not allowed')
    attn_rows = attn_word_attention_row(word, sent, trim=True, merge=True, first=True)
    sent_len = len(sym_better_tokenizer(sent,trim=True)['trim_merge_tokens'])
    word_pos = prop_word_position_in_senet(word, sent,trim=True,merge=True,first=True)
    attn_dis = torch.zeros(12,12,len(word_pos))
    attn_pos = []
    for lay in range(12):
        for hd in range(12):
            for occur in range(len(word_pos)):
                max_pos = torch.argmax(attn_rows[lay][hd][occur]).item()
                attn_dis_one_head = abs(max_pos-word_pos[occur])
                attn_dis[lay][hd][occur] = attn_dis_one_head
    if first == False:
        if absolute == True:
            res = attn_dis
        else:
            res = res/sent_len
    else:
        if absolute == True:
            res = attn_dis[:,:,0]
        else:
            res = res/sent_len
    return res

def prop_word_most_attend_position(word,sent,first=False): # 20221018214638
    '''获得一个词在一句话中的最关注距离, 只考虑trim_scale和merge后的情况, 第一个位置是0'''
    # [输入]
        # first开关: 是否值考虑一个词的第一个出现
    # [输出]
        # (tensor) 一个词在一句话中 在所有head中的最关注位置
    # [依赖]
        # attn_word_attention_row
        # prop_word_position_in_senet
    assert len(word.split(sep=' ')) == 1, print('\n[prop_word_most_attend_position]\nmore than one word not allowed')
    attn_rows = attn_word_attention_row(word, sent, trim=True, merge=True, first=True)
    word_pos = prop_word_position_in_senet(word, sent,trim=True,merge=True,first=True)
    attn_pos = torch.zeros(12,12,len(word_pos))
    for lay in range(12):
        for hd in range(12):
            for occur in range(len(word_pos)):
                max_pos = torch.argmax(attn_rows[lay][hd][occur]).item()
                attn_pos[lay][hd][occur] = max_pos
    if first == False:
        res = attn_pos
    else:
        res = attn_pos[:,:,0]
    return res

def stat_hidden_states_norm(sent,plot=False): # 20221013142939
    '''bert的13层hidden_states的范数的均值和标准差分布, 均值是一句话中每个词的范数的均值'''
    # [输入]
        # plot开关: 是否显示可视化
    # [输出]
        # (dict)各层的hidden state的范数的均值和标准差
    # [依赖]
        # pipe_pipeline
        # bert_tokenizer
    import statistics
    import matplotlib.pyplot as plt
    model_output = pipe_pipeline(sent)
    tokenized_sent = bert_tokenizer(sent)
    tokenized_tokens = bert_tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'])
    hidden_states_all_layers = model_output['hidden_states']
    layer_cnt = 0
    avg_norm_layers = []
    std_norm_layers = []
    for hidden_states_layer in hidden_states_all_layers:
        # print('Layer: '+ str(layer_cnt))
        # print('--------------------------')
        hidden_states_sent = hidden_states_layer[0]
        token_cnt = 0
        layer_norms = []
        for hidden_states_token in hidden_states_sent:
            # print('\t'+tokenized_tokens[token_cnt] +' '+ str(torch.norm(hidden_states_token).item()))
            layer_norms.append(torch.norm(hidden_states_token).item())
            token_cnt += 1
        layer_cnt += 1
        avg_norm = sum(layer_norms)/len(tokenized_tokens)
        std_norm = statistics.stdev(layer_norms)
        avg_norm_layers.append(avg_norm)
        std_norm_layers.append(std_norm)
        # print('Avg of Norms: '+ str(avg_norm))
        # print('Stdev of Norms: ' + str(std_norm))
        # print('--------------------------')
    if plot == True:
        fig = plt.figure()
        axes = fig.subplots(1,2)
        xs = [x for x in range(len(avg_norm_layers))]
        axes0 = axes[0].plot(avg_norm_layers,'x-b')
        for i,j in zip(xs,avg_norm_layers):
            axes[0].annotate('%.4f'%(j),xy=(i,j))
        axes[0].set_title('Average Hidden States Norm Across Layers')
        axes[0].set_xticks(range(len(avg_norm_layers)))
        axes1 = axes[1].plot(std_norm_layers,'x-b')
        for k,l in zip(xs,std_norm_layers):
            axes[1].annotate('%.4f'%(l),xy=(k,l))
        axes[1].set_title('Stdev of Hidden States Norm Across Layers')
        axes[1].set_xticks(range(len(std_norm_layers)))
        plt.show()
    return {'avg_norm_layers':avg_norm_layers,'std_norm_layers':std_norm_layers}

def prop_word_position_in_senet(word,sent,trim=True,merge=True,first=True): # 20221017211829
    '''返回一个词在句中的位置, 有11种情况, 用三个开关选择情况\ntrim开关: 句子是否削去句子两端的[CLS]和[SEP]\nmerge开关: 句子是否合并wordpiece\nfirst开关: 当词是wordpiece组合词时, 返回第一个词的位置还是所有位置, 仅当merge开关为false的情况生效, 仅对wordpiece组合词有意义'''
    # [输入值]
        # trim开关: 是否削去句子两端的[CLS]和[SEP]
        # merge开关: 是否合并wordpiece
        # first开关: 当词是wordpiece组合词时, 返回第一个词的位置还是所有位置, 仅当merge开关为false的情况生效, 仅对wordpiece组合词有意义
    # [返回值]
        # 位置列表, 元素为整数
    # [依赖]
        # sym_better_tokenizer
        # bert_tokenizer
    # [被依赖]
        # hidden_word_hidden_states_in_sent(word,sent,mode='first'): # 20221013165325
        # viz_hist_word_context_vecs(sent,word): # 20221013142738
        # get_word_attn_rows(word,sent_len,save_path): # 20221013142842
    # [备注]
        # 代码设计图 NOTION: 20221018160815
    assert len(word.split(sep=' ')) == 1, print('more than one word not allowed')
    
    res_better_tokenizer = sym_better_tokenizer(sent)
    notrim_nomerge_token_ids = res_better_tokenizer['notrim_nomerge_token_ids']
    notrim_nomerge_tokens = res_better_tokenizer['notrim_nomerge_tokens']
    notrim_merge_tokens = res_better_tokenizer['notrim_merge_tokens']
    trim_nomerge_tokens = res_better_tokenizer['trim_nomerge_tokens']
    trim_merge_tokens = res_better_tokenizer['trim_merge_tokens']

    is_simplex = len(bert_tokenizer(word)['input_ids']) == 3 # 是否为单纯词(即不被拆成word-piece的词)
    is_oov = bert_tokenizer.convert_tokens_to_ids(word) == 100

    if is_simplex:
        if is_oov:
            pos = []
            print('\n[prop_word_position_in_senet]\noov word, ignored')
        else:
            if trim == True:
                if merge == True:
                    pos = [idx for idx,value in enumerate(trim_merge_tokens) if value == word]
                else:
                    pos = [idx for idx,value in enumerate(trim_nomerge_tokens) if value == word]
            else:
                if merge == True:
                    pos = [idx for idx,value in enumerate(notrim_merge_tokens) if value == word]
                else:
                    pos = [idx for idx,value in enumerate(notrim_nomerge_tokens) if value == word]
    else:
        wordpiece_ids = bert_tokenizer(word)['input_ids'][1:-1]
        if trim == True:
            if merge == True:
                pos = [idx for idx,value in enumerate(trim_merge_tokens) if value == word]
            else:
                if first == True:
                    pos = []
                    for i in range(len(notrim_nomerge_token_ids)-len(wordpiece_ids)+1):
                        if wordpiece_ids == notrim_nomerge_token_ids[i:len(wordpiece_ids)+i].tolist():
                            pos.append(i-1)
                else:
                    pos = []
                    for i in range(len(notrim_nomerge_token_ids)-len(wordpiece_ids)+1):
                        if wordpiece_ids == notrim_nomerge_token_ids[i:len(wordpiece_ids)+i].tolist():
                            pos += [i+offset-1 for offset in range(len(wordpiece_ids))]
        else:
            if merge == True:
                pos = [idx for idx,value in enumerate(notrim_merge_tokens) if value == word]
            else:
                if first == True:
                    pos = []
                    for i in range(len(notrim_nomerge_token_ids)-len(wordpiece_ids)+1):
                        if wordpiece_ids == notrim_nomerge_token_ids[i:len(wordpiece_ids)+i].tolist():
                            pos.append(i)
                else:
                    pos = []
                    for i in range(len(notrim_nomerge_token_ids)-len(wordpiece_ids)+1):
                        if wordpiece_ids == notrim_nomerge_token_ids[i:len(wordpiece_ids)+i].tolist():
                            pos += [i+offset for offset in range(len(wordpiece_ids))]                   

    return pos

def stat_word_hidden_norm(word,sent): # 20221013142945
    '''计算一个词(的第一个出现)在一句话中的13个范数的均值'''
    tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
    input_ids = tokenized_sent['input_ids'][0].tolist()
    word_idx = bert_tokenizer.convert_tokens_to_ids(word)
    try:
        word_pos_1 = input_ids.index(word_idx)
        output = bert_model(**tokenized_sent,output_hidden_states=True)
        hidden_states = output['hidden_states']
        word_norms = []
        for layer in hidden_states:
            word_norms.append(torch.norm(layer[0][word_pos_1]).item())
            stat_word_hidden_norm = sum(word_norms)/len(word_norms)
        return stat_word_hidden_norm 
    except ValueError:
        return None

def stat_word_hidden_norm_in_sents(word,sents): # 20221013142950
    '''计算一个词在众多句子中的词向量的范数'''
    res = dict()
    sent_cnt = 0
    for sent in sents:
        mwn = stat_word_hidden_norm(word,sent)
        if mwn != None:
            res[sent_cnt] = mwn
        sent_cnt += 1
    return res

# viz_组
def viz_hist_word_hidden_states(sent,word,merge=False,first=True): # 20221013142738
    ''' 返回一句话中 一个词 在13个层的hidden_states, 并可视化为直方图'''
    # [输入]
        # merge开关: wordpiece组合型词是否合并
        # first开关: wordpiece组合型词取第一个词作为代表, 还是合并为一个hidden state
    # [输出]
        # 可视化图像
    # [依赖]
        # 依赖函数 hidden_word_hidden_states_in_sent
        # 依赖函数 sym_better_tokenizer
        # 依赖函数 prop_word_position_in_senet
    word_hiddens = hidden_word_hidden_states_in_sent(word,sent,merge=merge,first=first).permute(1,0,2)
    res_better_tokenizer = sym_better_tokenizer(sent,trim=False)
    tokenized_tokens = res_better_tokenizer['notrim_merge_tokens']
    word_pos = prop_word_position_in_senet(word,sent,trim=False,merge=merge,first=first)
    pos_cnt = 0
    for occur in word_hiddens.detach().numpy():
        fig = plt.figure(figsize=(20,20))
        axes = fig.subplots(3,5)
        axes = axes.ravel()
        for laycnt in range(13):
            axes[laycnt].hist(occur[laycnt],bins=48,edgecolor='k',density=True)
            axes[laycnt].set_xlim(-2,2)
            axes[laycnt].set_title(f'{word}, layer-{str(laycnt)},#{str(word_pos[pos_cnt])}')
            laycnt +=1
        axes.reshape(3,5)
        fig.suptitle(f'{sent}\n{tokenized_tokens}\npage {pos_cnt+1} of {len(word_pos)}')
        plt.show()
        pos_cnt += 1
    return

def viz_barplot_attn_row(word,sent,layer,head): # 20221018205503
    '''可视化一个词在一个head的attention分布'''
    attn_rows = attn_word_attention_row(word, sent,trim=True,merge=True,first=True)
    labels = sym_better_tokenizer(sent,trim=True)['trim_merge_tokens']
    for occur in attn_rows[layer][head]:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        xs = [i for i in range(len(occur))]
        ys = occur.detach().numpy()
        draw = ax.bar(xs,ys,width=1,edgecolor='k',align='center')
        ax.set_title(word)
        xticks = ax.set_xticks(xs)
        xlabels = ax.set_xticklabels(labels,rotation='vertical')
        plt.show()
    return

def viz_scatter_bert_preemb(lablist): # 20221013142953
    ''' 可视化BERT预训练词向量的空间分布, 输入为一个普通词表 '''
    labs = lablist[:]
    random.shuffle(labs)
    labids = bert_tokenizer.convert_tokens_to_ids(labs)
    word_embeddings_all = bert_model.embeddings.word_embeddings.weight
    word_embs = [word_embeddings_all[labid].detach().numpy() for labid in labids]
    def dim_reduce(word_embs):
    # 将300维的词向量降维2维
        ndim = 2
        # 实例化TSNE器
        tsne = TSNE(n_components=ndim,random_state=0)
        # 降维
        dim_reduced_vecs = tsne.fit_transform(word_embs)
        # 降维后的二维向量的横坐标
        x_vals = [v[0] for v in dim_reduced_vecs]
        # 降维后的二维向量的纵坐标
        y_vals = [v[1] for v in dim_reduced_vecs]
        return x_vals,y_vals
    # 为可视化准备横/纵坐标
    xs,ys = dim_reduce(word_embs)
    # matplotlib二维作图标准流程
    fig = plt.figure()
    ax = fig.subplots()
    # 散点图
    ax.scatter(xs,ys)
    # 给散点加标签
    cnt = 0
    for i in labs:
        axes = ax.annotate(labs[cnt],(xs[cnt],ys[cnt]))
        cnt += 1
    plt.show()
    return

# pipe组
def pipe_pipeline(sent): # 20221013142729
    '''傻瓜式的pipeline, 输入句子\n直接得到各层hidden states(1+12个)和各层attention矩阵(12×12)个\n输出为字典,结果全为tensor格式'''
    # 依赖全局对象bert_model_20221013172250
    # 依赖全局对象bert_tokenizer_20221013172247
    # 被依赖
        # stat_hidden_states_norm(sent,plot=False): # 20221013142939
        # get_sent_represent_vec_avg(sent): # 20221014105633
        # sym_better_tokenizer(sent): # 20221013142808
    tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
    output = bert_model(**tokenized_sent,output_hidden_states=True,output_attentions=True)
    attentions = torch.squeeze(torch.stack(output['attentions'],dim=0))
    hidden_states = torch.squeeze(torch.stack(output['hidden_states'],dim=0))
    last_hidden_state = torch.squeeze(output['last_hidden_state'])
    result = {'attentions':attentions,'hidden_states':hidden_states,'last_hidden_state':last_hidden_state}
    print('\n[pipe_pipeline]\nresult structure:')
    for i in result:
        print(f'{i}: {type(result[i])}')
    return result








