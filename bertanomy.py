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

logging.set_verbosity_error()

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_word_preemb(word):
    '''获得一个词的bert预训练词向量'''
    wordidx = bert_tokenizer.convert_tokens_to_ids(word)
    we = bert_model.embeddings.word_embeddings.weight
    word_preemb = we[wordidx]
    return word_preemb

def show_manual_bert_model_output():
    '''BERT输出数据结构手册'''
    print('''
bert_model_output:
    (odict: [4])
        0 (tensor: [1,句长,768]) last_hidden_state
        1 (tensor: [1, 768]) pooler output
        2 (tuple: [13]) hidden_states
            2.0 (tensor: [1,句长,768])
            2.1 (tensor: [1,句长,768])
                ...
            2.12 (tensor: [1,句长,768])
        3 (tuple: [13]) attentions
            3.0 (tensor: [1,12,句长,句长])
            3.1 (tensor: [1,12,句长,句长])
                ...
            3.12(tensor: [1,12,句长,句长])
        ''')
    return

def show_manual_bert_tokenizer_output():
    # BERT的Tokenizer的输出的数据结构的手册
    print('''
tokenized_sent:
    (dict: [3])
        (list: [≈句长+2]) input_ids
        (list: [≈句长+2]) token_type_ids
        (list: [≈句长+2]) attention_mask
        ''')
    return

def attn_trim(attn_matrix):
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

def get_all_interm_output(text):
    # 自动获得句子的上下文词向量矩阵(12+1个)和attention矩阵(144个)
    tokenized_text = bert_tokenizer(text,return_tensors='pt')
    output = bert_model(**tokenized_text,output_hidden_states=True,output_attentions=True)
    tokenized_tokens = dict(enumerate(bert_tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0])))
    print(tokenized_tokens)
    print(text)
    print(output.keys())
    return output

def compact_hidden_state(output_hidden_states):
    # 挤掉输出结果中维度为1的轴向
    hidden_states = torch.squeeze(torch.stack(output_hidden_states,dim=0))
    return hidden_states

def viz_hist_word_context_vecs(sent_hidden_states,word_pos):
    ''' 返回一句话中目标词在13个层的hidden_states'''
    # 参数1: BERT_MODEL返回值的['hidden_states']
    # 参数2: 目标词的位置
    hst = sent_hidden_states.permute(1,0,2)
    preplot = hst[word_pos].detach().numpy()
    fig = plt.figure(figsize=(20,20))
    axes = fig.subplots(3,5)
    axes = axes.ravel()
    laycnt = 0
    while laycnt<=12:
        axes[laycnt].hist(preplot[laycnt],bins=48,edgecolor='k',density=True)
        axes[laycnt].set_xlim(-2,2)
        laycnt +=1
    axes.reshape(3,5)
    plt.show()

def get_most_similar_preemb_by_vec(tensor_vec,tensor_target_vecs,topn):
    ''' 寻找一个BERT词向量和BERT预训练词向量中最相似的那个'''
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

def get_word_context_vec_in_sent(word,layer,sent):
    ''' 获得一个词(的第一个出现)在一句上下文中指定层的Context Vector'''
    # 依赖全局对象bert_tokenizer
    # 依赖全局对象bert_model
    tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
    sent_output = bert_model(**tokenized_sent,output_hidden_states=True)
    word_idx = bert_tokenizer.convert_tokens_to_ids(word)
    # 句子的input_ids序列
    sent_input_ids = tokenized_sent['input_ids'][0]
    tokens_sent = bert_tokenizer.convert_ids_to_tokens(sent_input_ids)
    # 目标词的第一个出现在input_ids中的位置
    wordpos = (sent_input_ids == word_idx).nonzero()[0][0].item()
    sent_cvecs = sent_output.hidden_states
    target_layer_sent_cvec = sent_cvecs[layer][0][wordpos]
    return target_layer_sent_cvec

def get_word_cvec_pos_in_sent(word,sent,layer):
    '''获得指定词在指定句在指定层的context vector'''
    # 依赖get_word_context_vec_in_sent
    word_cvec = get_word_context_vec_in_sent(word,layer,sent).detach()
    word_idx = bert_tokenizer.convert_tokens_to_ids(word)
    tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
    input_ids = tokenized_sent['input_ids'][0]
    word_pos = (input_ids == word_idx).nonzero()[0][0].item()
    return (word_cvec,word_pos)

def get_word_cvec_pos_in_sent_batch(word,sents,layer):
    '''get_word_cvec_pos_in_sent的批量句子版'''
    # 依赖get_word_cvec_pos_in_sent
    from time import time
    sents_len = len(sents)
    sents_cvec_ctn = torch.zeros(sents_len,768)
    sents_word_pos_ctn = torch.zeros(sents_len)
    sent_cnt = 0
    time_rec = time()
    for sent in sents:
        pair = get_word_cvec_pos_in_sent(word,sent,layer)
        sents_cvec_ctn[sent_cnt] = pair[0]
        sents_word_pos_ctn[sent_cnt] = pair[1]
        if sent_cnt%1000 == 0:
            print(f'processed {sent_cnt} sents')
            print(f'time cost: {time()-time_rec}')
            time_rec = time()
        sent_cnt += 1
    return sents_cvec_ctn,sents_word_pos_ctn

def get_word_cvec_pos_in_sent_batch_layer(word,sents):
    import pickle
    for layer in range(1,13):
        layer_res = get_word_cvec_pos_in_sent_batch(word,sents,layer)
        with open(f'd:/word_cvec_pos_layer_{layer}.pkl',mode='wb') as savefile:
            pickle.dump(layer_res,savefile)
        del layer_res
    return

# def get_id_map(sent):
#     '''获得word-piece和合并tokenization的映射表, 有list和dict两个版本'''
#     # tokenized句子
#     tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
#     # 获得tokenized句子的input_id串 '101 2065 3999 2837 ... 102'
#     input_ids = tokenized_sent['input_ids'][0]
#     # 截掉句子input_id两端的[101] 和 [102]
#     trimmed_input_ids = input_ids[1:-1]
#     # 截掉句子两端[CLS]和[SEP]后的tokenization结果
#     input_tokens = bert_tokenizer.convert_ids_to_tokens(trimmed_input_ids)

#     # 获得token-word合并映射表
#     # 获得token-word合并映射表:list版
#     cnt = -1
#     idmap_list = [] 
#     for tokenid in trimmed_input_ids: # token-word合并映射表 list
#         if bert_tokenizer.convert_ids_to_tokens([tokenid])[0].startswith('##'):
#             cnt = cnt
#         else:
#             cnt += 1
#         idmap_list.append(cnt)
#     # 获得token-word合并映射表:dict版
#     idmap_dict = dict()
#     for wordid in idmap_list:
#         idmap_dict[wordid] = [tokenid for tokenid,realwordid in enumerate(idmap_list) if realwordid==wordid]
#     return {'idmap_list': idmap_list, 'idmap_dict': idmap_dict}

def better_tokenizer(sent):
    '''获取正常的文字tokenization结果, wordpiece版和合并版'''
    # 依赖get_id_map()
    # 依赖get_onepiece_tokens()
    # 依赖全局对象bert_tokenizer
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

def get_actual_len_sents(sents,sent_len):
    '''返回句子中, bert分词后仍为指定长度的句子'''
    from thehow import bertology
    actual_len_sents = []
    sent_cnt = 0
    for sent in sents:
        if len(bertology.better_tokenizer(sent)['dense_tokens']) == sent_len:
            actual_len_sents.append(sent)
            sent_cnt +=1
            if sent_cnt%100 ==0:
                print(f'processed {sent_cnt} sents')
    return actual_len_sents


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
    better_token = better_tokenizer(sent)
    token_word_id_map_list = better_token['idmap_list']
    token_word_id_map_dict = better_token['idmap_dict']
    
    # 获得目标词在BERT词表中的编号
    target_word_id = bert_tokenizer.convert_tokens_to_ids(word)
    # 获得目标词在token-word合并映射表转换前的attention矩阵中的行号
    target_word_attn_row_num_token =  (input_ids == target_word_id).nonzero()[0]
    # 获得目标词在token-word合并映射表转换后的attention矩阵中的行号
    target_word_attn_row_num_dense = token_word_id_map_list[target_word_attn_row_num_token]

    # 获得未经裁边和压缩的attention矩阵
    attn = bert_model(**tokenized_sent,output_attentions=True)['attentions'][attn_layer][0][attn_head].detach()
    # 获得裁边后的attention矩阵
    attn_trimmed = attn_trim(attn) # 修边attn矩阵
    # 获得裁边并压缩后的attention矩阵(依据token-word合并映射表)
    attn_dense = attn_matrix_denser(token_word_id_map_dict,attn_trimmed) # 修边-合并矩阵
    # 获得目标词在裁边并压缩后attention矩阵中对应的行(tensor)
    word_attn_dense_row = attn_dense[target_word_attn_row_num_dense].tolist()
    # 获得目标词在裁边并压缩后attention矩阵中对应的行(dict: key是位置, value是关注度)
    #word_attn_row_dict = dict(enumerate(word_attn_row.tolist()))
    return (word_attn_dense_row,attn_dense,target_word_attn_row_num_dense)

def get_word_dense_attn_rows_one_sent(word:str,sent:str):
    '''获取一个词在一句话中的attention row(压缩版)'''
    # tokenized句子
    tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
    # 获得tokenized句子的input_id串 '101 2065 3999 2837 ... 102'
    input_ids = tokenized_sent['input_ids'][0]
    # 截掉句子input_id两端的[101] 和 [102]
    input_ids = input_ids[1:-1]
    # 截掉句子两端[CLS]和[SEP]后的tokenization结果
    input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

    # 获得token-word合并映射表
    better_token = better_tokenizer(sent)
    token_word_id_map_list = better_token['idmap_list']
    token_word_id_map_dict = better_token['idmap_dict']
    
    # 获得目标词在BERT词表中的编号
    target_word_id = bert_tokenizer.convert_tokens_to_ids(word)
    # 获得目标词在token-word合并映射表转换前的attention矩阵中的行号
    target_word_attn_row_num_token =  (input_ids == target_word_id).nonzero()[0]
    # 获得目标词在token-word合并映射表转换后的attention矩阵中的行号
    target_word_attn_row_num_dense = token_word_id_map_list[target_word_attn_row_num_token]

    # 获得实际句长
    actual_sent_len = len(token_word_id_map_dict)

    # 容器: dense attention矩阵(144个)
    attn_dense_matrices = torch.zeros(12,12,actual_sent_len,actual_sent_len)
    # 容器: dense attention行(144个)
    attn_dense_rows = torch.zeros(12,12,actual_sent_len)

    # 获得未经裁边和压缩的attention矩阵
    raw_attns = bert_model(**tokenized_sent,output_attentions=True)['attentions']

    for layer in range(12):
        for head in range(12):
            raw_attn = raw_attns[layer][0][head].detach()
            # 获得裁边后的attention矩阵
            attn_trimmed = attn_trim(raw_attn) # 修边attn矩阵
            # 获得裁边并压缩后的attention矩阵(依据token-word合并映射表)
            attn_dense = attn_matrix_denser(token_word_id_map_dict,attn_trimmed) # 修边-合并矩阵
            attn_dense_matrices[layer][head] = attn_dense
            # 获得目标词在裁边并压缩后attention矩阵中对应的行(tensor)
            word_attn_dense_row = attn_dense[target_word_attn_row_num_dense].detach()
            attn_dense_rows[layer][head]=word_attn_dense_row
            # 获得目标词在裁边并压缩后attention矩阵中对应的行(dict: key是位置, value是关注度)
            #word_attn_row_dict = dict(enumerate(word_attn_row.tolist()))
    return {'matrices':attn_dense_matrices,'rows':attn_dense_rows}

def get_word_attn_rows(word,sent_len,save_path):
    '''获得一个词对应的attention矩阵中的一行'''
    from pathlib import Path
    import pickle
    from thehow import posdist
    sents = posdist.sent_selector(word,sent_len,posdist.en_corpus)
    sents += posdist.sent_selector(word,sent_len-1,posdist.en_corpus)
    sents += posdist.sent_selector(word,sent_len-2,posdist.en_corpus)
    sents += posdist.sent_selector(word,sent_len-3,posdist.en_corpus)
    print(f'No. of sents: {len(sents)}')
    sents_true_len = []
    word_pos = []
    for sent in sents:
        # tokenized句子
        res_better_token  = better_tokenizer(sent)
        true_len  = len(res_better_token['dense_tokens'])
        if true_len == sent_len:
            sents_true_len.append(sent)
    print(f'No. of sents of len {sent_len}: {len(sents_true_len)}')
    len_sents_true_len = len(sents_true_len)
    attn_rows = torch.zeros(12,12,len_sents_true_len,20)
    tsent_cnt = 0
    for tsent in sents_true_len:
        tokenized_sent = bert_tokenizer(tsent,return_tensors='pt')
        res_better_token  = better_tokenizer(tsent)
        idmap_dict = res_better_token['idmap_dict']
        idmap_list = res_better_token['idmap_list']
        sent_attn_144 = torch.stack(bert_model(**tokenized_sent,output_attentions=True)['attentions']).squeeze()
        word_row_num = get_word_attn_row_num(word,tsent)
        word_pos.append(word_row_num)
        for layer in range(12):
            for head in range(12):
                attn_trimmed = attn_trim(sent_attn_144[layer][head])
                attn_dense = attn_matrix_denser(idmap_dict,attn_trimmed)
                attn_row = attn_dense[word_row_num].detach()
                attn_rows[layer][head][tsent_cnt]=attn_row
        tsent_cnt += 1
        if tsent_cnt%100 ==0:
            print(f'processed {tsent_cnt} sents')
    savename_rows = Path(save_path).joinpath(f'attn_rows_{word}_{sent_len}.pkl')
    savename_labs = Path(save_path).joinpath(f'attn_labs_{word}_{sent_len}.pkl')
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

def attn_matrix_denser(idmap_dict,trimmed_attn_matrix):
    '''压缩attention矩阵'''
    # 先把行压缩
    def rebuild_row(idmap_dict,trimmed_attn_matrix):
        new_tensor = torch.zeros((len(idmap_dict.keys()),len(trimmed_attn_matrix)))
        for row_no_new in idmap_dict:
            new_tensor[row_no_new] = trimmed_attn_matrix[idmap_dict[row_no_new]].sum(axis=0)/len(idmap_dict[row_no_new])
        return new_tensor
    # 再把列压缩
    def rebuild_col(idmap_dict,trimmed_attn_matrix):
        new_tensor = torch.zeros((len(idmap_dict.keys()),len(idmap_dict.keys())))
        for col_no_new in idmap_dict:
            new_tensor[:,col_no_new] = trimmed_attn_matrix[:,idmap_dict[col_no_new]].sum(axis=1)
        return new_tensor
    row_proced = rebuild_row(idmap_dict,trimmed_attn_matrix)
    col_proced = rebuild_col(idmap_dict,row_proced)
    return col_proced

def get_word_attn_row_num(word,sent):
    '''获得一个词在句中的位置(dense attn matrix的行号)'''
    # tokenized句子
    tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
    # 获得tokenized句子的input_id串 '101 2065 3999 2837 ... 102'
    input_ids = tokenized_sent['input_ids'][0]
    # 截掉句子input_id两端的[101] 和 [102]
    input_ids = input_ids[1:-1]

    # 获得token-word合并映射表
    better_token = better_tokenizer(sent)
    token_word_id_map_list = better_token['idmap_list']
    token_word_id_map_dict = better_token['idmap_dict']
    
    # 获得目标词在BERT词表中的编号
    target_word_id = bert_tokenizer.convert_tokens_to_ids(word)
    # 获得目标词在token-word合并映射表转换前的attention矩阵中的行号
    target_word_attn_row_num_token =  (input_ids == target_word_id).nonzero()[0]
    # 获得目标词在token-word合并映射表转换后的attention矩阵中的行号
    target_word_attn_row_num_dense = token_word_id_map_list[target_word_attn_row_num_token]

    return target_word_attn_row_num_dense

def attndistance(word,sent,attn_layer,attn_head):
    '''计算一个词在一个layer,一个head,一个句子中的关注距离'''
    # 依赖get_word_attnpos()
    attnpos_res = get_word_attnpos(word,sent,attn_layer,attn_head)
    attnpos_dist = attnpos_res[0]
    taraget_word_idx = attnpos_res[3]
    attnpos_max_idx = attnpos_dist.index(max(attnpos_dist))
    distance = abs(taraget_word_idx-attnpos_max_idx)
    return distance

def attndistance_all(word,sents):
    '''计算一个词在所有句子中, 12个layer, 12个head的全部关注距离'''
    # 依赖attndistance
    perlayer = []
    for lay in range(12):
        perhead = []
        for head in range(12):
            persent = []
            for sent in sents:
                attndis = attndistance(word,sent,lay,head)
                persent.append(attndis)
            perhead.append(persent)
        perlayer.append(perhead)
    return perlayer


def attnpos_batch(word:str,sents:list):
    '''所有layer和head中, 获取n句话中一个词对各个位置的关注, 返回值为字典'''
    # attnpos 处理一个head需要0.18s. 那么, 20句话,144个head约需要9.6分钟
    attnpos_layers = []
    for layer in range(0,12):
        attnpos_heads = []
        for head in range(0,12):
            attnpos_sents = []
            for sent in sents:
                attnpos_sent = get_word_attnpos(word,sent,layer,head)[0]
                attnpos_sents.append(attnpos_sent)
            attnpos_heads.append(attnpos_sents)
        attnpos_layers.append(attnpos_heads)
    # 返回值
    #  list 12 Layer号
    #    list 12 Head号
    #       list 20 句子号
    return attnpos_layers

def attnpos_stat(attnpos_batch_result:list):
    '''输入为attnpos_batch的输出, 汇报各层和各head的平均关注位置分布'''
    layers_report = []
    for layer in attnpos_batch_result:
        heads_report = []
        for head in layer:
            sent_ctn = []
            for sent in head:
                sent_ctn.append(sent)
            head_report = [sum(i)/len(attnpos_batch_result[0][0]) for i in zip(*sent_ctn)]
            heads_report.append(head_report)
        layers_report.append(heads_report)
    # 返回值
    #  list 12 Layer号
    #    list 12 Head号
    return layers_report



def attn_matrix_denser_batch(sent):
    '''将一句话的144个,attention的token-token关注矩阵按照对照表压缩成词-词关注矩阵'''
    # 依赖全局变量bert_model
    # tokenizr句子
    tokenization = better_tokenizer(sent)
    tokenized_sent = tokenization['tokenized_sent']
    # 获得token-word合并映射表
    idmap_dict = tokenization['idmap_dict']
    
    # 获得未经裁边和压缩的attention矩阵
    attn144 = bert_model(**tokenized_sent,output_attentions=True)['attentions']
    # 裁边的attention矩阵
    attn144_trim = torch.zeros(12,12,len(attn144[0][0][0])-2,len(attn144[0][0][0])-2)
    # 裁边和压缩的attention矩阵
    attn144_trim_dense = torch.zeros(12,12,len(idmap_dict),len(idmap_dict))
    for lay in range(12):
        for head in range(12):
            attn144_trim[lay][head] = attn_trim(attn144[lay][0][head].detach())
            attn144_trim_dense[lay][head] = attn_matrix_denser(idmap_dict,attn144_trim[lay][head])
    return attn144_trim_dense

def mean_attn_distance_rel_one_sent_all_head(sent):
    '''一个句子144个attention head的平均关注距离'''
    # 依赖attn_denser
    heads_ctn = []
    attn_dense = attn_denser(sent)[0].tolist()
    for lay in range(12):
        for head in range(12):
            attndis_row_ctn = []
            row_cnt = 0
            for row in attn_dense[lay][head]:
                rowmaxid = row.index(max(row))
                rowid = row_cnt
                row_attndis = abs(rowmaxid-rowid)
                attndis_row_ctn.append(row_attndis)
                row_cnt += 1
            heads_ctn.append(sum(attndis_row_ctn)/len(attndis_row_ctn))
    res_abs = sum(heads_ctn)/144
    res_rel = res_abs/len(attn_dense[0][0])
    return res_rel

def word_attn_distance(word,sent):
    '''一个词在一句话中的144个head的平均关注距离'''
    # 得知词在压缩矩阵的第几行
    attn_denser_res = attn_denser(sent)
    attn_dense = attn_denser_res[0]
    idmap = attn_denser_res[1]
    tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
    word_idx = bert_tokenizer.convert_tokens_to_ids(word)
    trim_input_ids = tokenized_sent['input_ids'][0][1:-1].tolist()
    trim_word_pos = trim_input_ids.index(word_idx)
    word_rowid = [i[0] for i in idmap.items() if trim_word_pos in i[1]][0]
    # 
    per_head_attn_dis_ctn = []
    for layer in range(12):
        for head in range(12):
            word_attn_row = attn_dense[layer][head][word_rowid].tolist()
            rowmaxid = word_attn_row.index(max(word_attn_row))
            row_attndis = abs(rowmaxid-word_rowid)
            per_head_attn_dis_ctn.append(row_attndis)
    mean_attn_dis = sum(per_head_attn_dis_ctn)/144
    sent_len = len(attn_dense[0][0])
    mean_attn_dis_rel = mean_attn_dis/sent_len
    return mean_attn_dis_rel

def word_attn_distance_sents(word,sents):
    ''' 指定词,在各个指定句子们的所有attention head(n×144)中, 分别的平均依存距离, 返回值是dict(), 记录每句话的结果'''
    #
    dis_ctn = dict()
    sent_cnt = 0
    for sent in sents:
        try:
            mean_attn_dis_rel = word_attn_distance(word,sent)
            dis_ctn[sent_cnt] = mean_attn_dis_rel
        except:
            pass
        sent_cnt += 1
    return dis_ctn

def hidden_states_norm(sent,plot=False):
    '''bert的13层hidden_states的范数的均值和标准差分布'''
    import statistics
    import matplotlib.pyplot as plt
    model_output = get_all_interm_output(sent)
    tokenized_sent = bert_tokenizer(sent)
    tokenized_tokens = bert_tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'])
    hidden_states_all_layers = model_output.hidden_states
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

def get_word_context_vecs_norm(word,sent):
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
            get_word_context_vecs_norm = sum(word_norms)/len(word_norms)
        return get_word_context_vecs_norm 
    except ValueError:
        return None

def mean_word_norm_sents(word,sents):
    '''计算一个词在众多句子中的词向量的范数'''
    res = dict()
    sent_cnt = 0
    for sent in sents:
        mwn = get_word_context_vecs_norm(word,sent)
        if mwn != None:
            res[sent_cnt] = mwn
        sent_cnt += 1
    return res

def viz_scatter_bert_preemb(lablist):
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


class glassbox_bert:
    '''将BERT的Embedings层和Encoder0层的所有中间变量输出可获得BERT的所有中间阶段表征, 包括QKV矩阵'''
    def __init__(self,text):
        self.intro_doc = '''
        执行顺序:
        bert_tokenized_text 
        input_ids 
        lay_emb_laynorm 
        vec_layer_0_out 
        vec_queries_12_head 
        lay_div_transpose 
        vec_queries_12_head_div 
        vec_keys_12_head 
        vec_keys_12_head_div 
        vec_values_12_head 
        vec_values_12_head_div 
        vec_qkt_12_head 
        vec_qkt_dk_12_head 
        lay_softmax 
        vec_softmax: Attention矩阵(12个), 每个矩阵逐行阅读(和为1)[句,头,行,列]
        vec_qktv: Context vectors (QK.TV)
        lay_selfout_dense 
        vec_selfout_dense 
        vec_selfout_add 
        lay_selfout_norm 
        vec_selfout_norm 
        lay_interm_dense 
        vec_interm_dense 
        vec_interm_gelu 
        lay_out_dense 
        vec_out_dense 
        vec_out_add 
        lay_out_norm 
        vec_out_norm: 第一层输出hidden_states
        '''
        print(self.intro_doc)
        self.text = text
        self.bert_tokenized_text = bert_tokenizer(self.text,return_tensors='pt')
        self.input_ids = self.bert_tokenized_text['input_ids'][0]
        # Embeddings
        self.lay_emb_laynorm = bert_model.embeddings.LayerNorm
        self.vec_layer_0_out = self.get_pre_embeddings()
        # BertSelfAttention
        # Q
        # div是指将768分成16*64
        self.lay_div_transpose = bert_model.encoder.layer[0].attention.self.transpose_for_scores
        self.vec_queries_12_head = bert_model.encoder.layer[0].attention.self.query(self.vec_layer_0_out)
        self.vec_queries_12_head_div = self.lay_div_transpose(self.vec_queries_12_head)
        # K
        self.vec_keys_12_head = bert_model.encoder.layer[0].attention.self.key(self.vec_layer_0_out)
        self.vec_keys_12_head_div = self.lay_div_transpose(self.vec_keys_12_head)
        # V
        self.vec_values_12_head = bert_model.encoder.layer[0].attention.self.value(self.vec_layer_0_out)
        self.vec_values_12_head_div = self.lay_div_transpose(self.vec_values_12_head)
        # QK.T
        self.vec_qkt_12_head = torch.matmul(self.vec_queries_12_head_div,self.vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.vec_qkt_dk_12_head = self.vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.lay_softmax = nn.Softmax(dim=-1)
        self.vec_softmax = self.lay_softmax(self.vec_qkt_dk_12_head)
        # Context Vectors (12个head的concatenate好的)
        self.vec_qktv = torch.matmul(self.vec_softmax, self.vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__new_context_vecs_shape = self.vec_qktv.shape[:-2] + (768,)
        self.vec_qktv = self.vec_qktv.view(*self.__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.lay_selfout_dense = bert_model.encoder.layer[0].attention.output.dense
        self.vec_selfout_dense = self.lay_selfout_dense(self.vec_qktv)
        # Add
        self.vec_selfout_add = self.vec_selfout_dense + self.vec_layer_0_out
        # LayerNorm
        self.lay_selfout_norm = bert_model.encoder.layer[0].attention.output.LayerNorm
        self.vec_selfout_norm = self.lay_selfout_norm(self.vec_selfout_add)

        # Sub-layer-2
        # BertIntermediate
        # Dense
        self.lay_interm_dense = bert_model.encoder.layer[0].intermediate.dense
        self.vec_interm_dense = self.lay_interm_dense(self.vec_selfout_norm)
        # Activation
        self.vec_interm_gelu = gelu(self.vec_interm_dense)
        # BertOutput
        # Dense
        self.lay_out_dense = bert_model.encoder.layer[0].output.dense
        self.vec_out_dense = self.lay_out_dense(self.vec_interm_gelu)
        # Add
        self.vec_out_add = self.vec_out_dense + self.vec_selfout_norm
        # LayerNorm
        self.lay_out_norm = bert_model.encoder.layer[0].output.LayerNorm
        self.vec_out_norm = self.lay_out_norm(self.vec_out_add)
        
    def get_pre_embeddings(self):
        preemb = torch.zeros(1,len(self.input_ids),768)
        tokenid = 0
        for token in self.input_ids:
            # 获得预训练词向量
            wordemb = bert_model.embeddings.word_embeddings.weight[token]
            # 获得预训练位置向量
            posemb = bert_model.embeddings.position_embeddings.weight[tokenid]
            # 获得预训练分句向量
            segemb = bert_model.embeddings.token_type_embeddings.weight[0]
            # 把词向量, 位置向量, 分句向量加起来
            sumemb = wordemb + posemb + segemb
            # 把和向量取LayerNorm
            laynormemb = self.lay_emb_laynorm(sumemb)
            preemb[0][tokenid] = laynormemb
            tokenid += 1
        return preemb

class glassbox_bert_12:
    '''可获得BERT的所有中间阶段表征, 包括QKV矩阵'''
    # Glass Box == Unit testing [CHURCH2021]
    # glassbox_bert12 舍弃PreEmbedding的BERT Encoder第1层
    # glassbox_bert12 鼻饲, 类比不经过嘴咀嚼, 直接将营养液(词向量)输入到bert
    # 返回值是一个对象, 对象的属性是各个阶段值
    # 不包括各Dropout层
    def __init__(self,vecs):
    # Layer 1
        # BertSelfAttention
       
        self.e1_input_vecs = vecs
        self.e1_lay_div_transpose = bert_model.encoder.layer[0].attention.self.transpose_for_scores
        # Q
        self.e1_vec_queries_12_head = bert_model.encoder.layer[0].attention.self.query(self.e1_input_vecs)
        self.e1_vec_queries_12_head_div = self.e1_lay_div_transpose(self.e1_vec_queries_12_head) # 这里的transpose不是转置, 而是将12个head拆开
        # K
        self.e1_vec_keys_12_head = bert_model.encoder.layer[0].attention.self.key(self.e1_input_vecs)
        self.e1_vec_keys_12_head_div = self.e1_lay_div_transpose(self.e1_vec_keys_12_head)  # 这里的transpose不是转置, 而是将12个head拆开
        # V
        self.e1_vec_values_12_head = bert_model.encoder.layer[0].attention.self.value(self.e1_input_vecs)
        self.e1_vec_values_12_head_div = self.e1_lay_div_transpose(self.e1_vec_values_12_head) # 这里的transpose不是转置, 而是将12个head拆开
        # QK.T
        self.e1_vec_qkt_12_head = torch.matmul(self.e1_vec_queries_12_head_div,self.e1_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e1_vec_qkt_dk_12_head = self.e1_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e1_lay_softmax = nn.Softmax(dim=-1)
        self.e1_vec_softmax = self.e1_lay_softmax(self.e1_vec_qkt_dk_12_head)
        # Context Vectors
        self.e1_vec_qktv = torch.matmul(self.e1_vec_softmax, self.e1_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e1__new_context_vecs_shape = self.e1_vec_qktv.shape[:-2] + (768,)
        self.e1_vec_qktv = self.e1_vec_qktv.view(*self.__e1__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e1_lay_selfout_dense = bert_model.encoder.layer[0].attention.output.dense
        self.e1_vec_selfout_dense = self.e1_lay_selfout_dense(self.e1_vec_qktv)
        # Add
        self.e1_vec_selfout_add = self.e1_vec_selfout_dense + self.e1_input_vecs
        # LayerNorm
        self.e1_lay_selfout_norm = bert_model.encoder.layer[0].attention.output.LayerNorm
        self.e1_vec_selfout_norm = self.e1_lay_selfout_norm(self.e1_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e1_lay_interm_dense = bert_model.encoder.layer[0].intermediate.dense
        self.e1_vec_interm_dense = self.e1_lay_interm_dense(self.e1_vec_selfout_norm)
        # Activation
        self.e1_vec_interm_gelu = gelu(self.e1_vec_interm_dense)
        # BertOutput
        # Dense
        self.e1_lay_out_dense = bert_model.encoder.layer[0].output.dense
        self.e1_vec_out_dense = self.e1_lay_out_dense(self.e1_vec_interm_gelu)
        # Add
        self.e1_vec_out_add = self.e1_vec_out_dense + self.e1_vec_selfout_norm
        # LayerNorm
        self.e1_lay_out_norm = bert_model.encoder.layer[0].output.LayerNorm
        self.e1_vec_out_norm = self.e1_lay_out_norm(self.e1_vec_out_add)

        self.e1_vec_layer_output = self.e1_vec_out_norm

    #Layer 2
        self.e2_input_vecs = self.e1_vec_layer_output
        self.e2_lay_div_transpose = bert_model.encoder.layer[1].attention.self.transpose_for_scores
        self.e2_vec_queries_12_head = bert_model.encoder.layer[1].attention.self.query(self.e2_input_vecs)
        self.e2_vec_queries_12_head_div = self.e2_lay_div_transpose(self.e2_vec_queries_12_head)
        # K
        self.e2_vec_keys_12_head = bert_model.encoder.layer[1].attention.self.key(self.e2_input_vecs)
        self.e2_vec_keys_12_head_div = self.e2_lay_div_transpose(self.e2_vec_keys_12_head)
        # V
        self.e2_vec_values_12_head = bert_model.encoder.layer[1].attention.self.value(self.e2_input_vecs)
        self.e2_vec_values_12_head_div = self.e2_lay_div_transpose(self.e2_vec_values_12_head)
        # QK.T
        self.e2_vec_qkt_12_head = torch.matmul(self.e2_vec_queries_12_head_div,self.e2_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e2_vec_qkt_dk_12_head = self.e2_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e2_lay_softmax = nn.Softmax(dim=-1)
        self.e2_vec_softmax = self.e2_lay_softmax(self.e2_vec_qkt_dk_12_head)
        # Context Vectors
        self.e2_vec_qktv = torch.matmul(self.e2_vec_softmax, self.e2_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e2__new_context_vecs_shape = self.e2_vec_qktv.shape[:-2] + (768,)
        self.e2_vec_qktv = self.e2_vec_qktv.view(*self.__e2__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e2_lay_selfout_dense = bert_model.encoder.layer[1].attention.output.dense
        self.e2_vec_selfout_dense = self.e2_lay_selfout_dense(self.e2_vec_qktv)
        # Add
        self.e2_vec_selfout_add = self.e2_vec_selfout_dense + self.e2_input_vecs
        # LayerNorm
        self.e2_lay_selfout_norm = bert_model.encoder.layer[1].attention.output.LayerNorm
        self.e2_vec_selfout_norm = self.e2_lay_selfout_norm(self.e2_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e2_lay_interm_dense = bert_model.encoder.layer[1].intermediate.dense
        self.e2_vec_interm_dense = self.e2_lay_interm_dense(self.e2_vec_selfout_norm)
        # Activation
        self.e2_vec_interm_gelu = gelu(self.e2_vec_interm_dense)
        # BertOutput
        # Dense
        self.e2_lay_out_dense = bert_model.encoder.layer[1].output.dense
        self.e2_vec_out_dense = self.e2_lay_out_dense(self.e2_vec_interm_gelu)
        # Add
        self.e2_vec_out_add = self.e2_vec_out_dense + self.e2_vec_selfout_norm
        # LayerNorm
        self.e2_lay_out_norm = bert_model.encoder.layer[1].output.LayerNorm
        self.e2_vec_out_norm = self.e2_lay_out_norm(self.e2_vec_out_add)

        self.e2_vec_layer_output = self.e2_vec_out_norm

    #Layer 3
        self.e3_input_vecs = self.e2_vec_layer_output
        self.e3_lay_div_transpose = bert_model.encoder.layer[2].attention.self.transpose_for_scores
        self.e3_vec_queries_12_head = bert_model.encoder.layer[2].attention.self.query(self.e3_input_vecs)
        self.e3_vec_queries_12_head_div = self.e3_lay_div_transpose(self.e3_vec_queries_12_head)
        # K
        self.e3_vec_keys_12_head = bert_model.encoder.layer[2].attention.self.key(self.e3_input_vecs)
        self.e3_vec_keys_12_head_div = self.e3_lay_div_transpose(self.e3_vec_keys_12_head)
        # V
        self.e3_vec_values_12_head = bert_model.encoder.layer[2].attention.self.value(self.e3_input_vecs)
        self.e3_vec_values_12_head_div = self.e3_lay_div_transpose(self.e3_vec_values_12_head)
        # QK.T
        self.e3_vec_qkt_12_head = torch.matmul(self.e3_vec_queries_12_head_div,self.e3_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e3_vec_qkt_dk_12_head = self.e3_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e3_lay_softmax = nn.Softmax(dim=-1)
        self.e3_vec_softmax = self.e3_lay_softmax(self.e3_vec_qkt_dk_12_head)
        # Context Vectors
        self.e3_vec_qktv = torch.matmul(self.e3_vec_softmax, self.e3_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e3__new_context_vecs_shape = self.e3_vec_qktv.shape[:-2] + (768,)
        self.e3_vec_qktv = self.e3_vec_qktv.view(*self.__e3__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e3_lay_selfout_dense = bert_model.encoder.layer[2].attention.output.dense
        self.e3_vec_selfout_dense = self.e3_lay_selfout_dense(self.e3_vec_qktv)
        # Add
        self.e3_vec_selfout_add = self.e3_vec_selfout_dense + self.e3_input_vecs
        # LayerNorm
        self.e3_lay_selfout_norm = bert_model.encoder.layer[2].attention.output.LayerNorm
        self.e3_vec_selfout_norm = self.e3_lay_selfout_norm(self.e3_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e3_lay_interm_dense = bert_model.encoder.layer[2].intermediate.dense
        self.e3_vec_interm_dense = self.e3_lay_interm_dense(self.e3_vec_selfout_norm)
        # Activation
        self.e3_vec_interm_gelu = gelu(self.e3_vec_interm_dense)
        # BertOutput
        # Dense
        self.e3_lay_out_dense = bert_model.encoder.layer[2].output.dense
        self.e3_vec_out_dense = self.e3_lay_out_dense(self.e3_vec_interm_gelu)
        # Add
        self.e3_vec_out_add = self.e3_vec_out_dense + self.e3_vec_selfout_norm
        # LayerNorm
        self.e3_lay_out_norm = bert_model.encoder.layer[2].output.LayerNorm
        self.e3_vec_out_norm = self.e3_lay_out_norm(self.e3_vec_out_add)

        self.e3_vec_layer_output = self.e3_vec_out_norm

    #Layer 4
        self.e4_input_vecs = self.e3_vec_layer_output
        self.e4_lay_div_transpose = bert_model.encoder.layer[3].attention.self.transpose_for_scores
        self.e4_vec_queries_12_head = bert_model.encoder.layer[3].attention.self.query(self.e4_input_vecs)
        self.e4_vec_queries_12_head_div = self.e4_lay_div_transpose(self.e4_vec_queries_12_head)
        # K
        self.e4_vec_keys_12_head = bert_model.encoder.layer[3].attention.self.key(self.e4_input_vecs)
        self.e4_vec_keys_12_head_div = self.e4_lay_div_transpose(self.e4_vec_keys_12_head)
        # V
        self.e4_vec_values_12_head = bert_model.encoder.layer[3].attention.self.value(self.e4_input_vecs)
        self.e4_vec_values_12_head_div = self.e4_lay_div_transpose(self.e4_vec_values_12_head)
        # QK.T
        self.e4_vec_qkt_12_head = torch.matmul(self.e4_vec_queries_12_head_div,self.e4_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e4_vec_qkt_dk_12_head = self.e4_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e4_lay_softmax = nn.Softmax(dim=-1)
        self.e4_vec_softmax = self.e4_lay_softmax(self.e4_vec_qkt_dk_12_head)
        # Context Vectors
        self.e4_vec_qktv = torch.matmul(self.e4_vec_softmax, self.e4_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e4__new_context_vecs_shape = self.e4_vec_qktv.shape[:-2] + (768,)
        self.e4_vec_qktv = self.e4_vec_qktv.view(*self.__e4__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e4_lay_selfout_dense = bert_model.encoder.layer[3].attention.output.dense
        self.e4_vec_selfout_dense = self.e4_lay_selfout_dense(self.e4_vec_qktv)
        # Add
        self.e4_vec_selfout_add = self.e4_vec_selfout_dense + self.e4_input_vecs
        # LayerNorm
        self.e4_lay_selfout_norm = bert_model.encoder.layer[3].attention.output.LayerNorm
        self.e4_vec_selfout_norm = self.e4_lay_selfout_norm(self.e4_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e4_lay_interm_dense = bert_model.encoder.layer[3].intermediate.dense
        self.e4_vec_interm_dense = self.e4_lay_interm_dense(self.e4_vec_selfout_norm)
        # Activation
        self.e4_vec_interm_gelu = gelu(self.e4_vec_interm_dense)
        # BertOutput
        # Dense
        self.e4_lay_out_dense = bert_model.encoder.layer[3].output.dense
        self.e4_vec_out_dense = self.e4_lay_out_dense(self.e4_vec_interm_gelu)
        # Add
        self.e4_vec_out_add = self.e4_vec_out_dense + self.e4_vec_selfout_norm
        # LayerNorm
        self.e4_lay_out_norm = bert_model.encoder.layer[3].output.LayerNorm
        self.e4_vec_out_norm = self.e4_lay_out_norm(self.e4_vec_out_add)

        self.e4_vec_layer_output = self.e4_vec_out_norm

    #Layer 5
        self.e5_input_vecs = self.e4_vec_layer_output
        self.e5_lay_div_transpose = bert_model.encoder.layer[4].attention.self.transpose_for_scores
        self.e5_vec_queries_12_head = bert_model.encoder.layer[4].attention.self.query(self.e5_input_vecs)
        self.e5_vec_queries_12_head_div = self.e5_lay_div_transpose(self.e5_vec_queries_12_head)
        # K
        self.e5_vec_keys_12_head = bert_model.encoder.layer[4].attention.self.key(self.e5_input_vecs)
        self.e5_vec_keys_12_head_div = self.e5_lay_div_transpose(self.e5_vec_keys_12_head)
        # V
        self.e5_vec_values_12_head = bert_model.encoder.layer[4].attention.self.value(self.e5_input_vecs)
        self.e5_vec_values_12_head_div = self.e5_lay_div_transpose(self.e5_vec_values_12_head)
        # QK.T
        self.e5_vec_qkt_12_head = torch.matmul(self.e5_vec_queries_12_head_div,self.e5_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e5_vec_qkt_dk_12_head = self.e5_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e5_lay_softmax = nn.Softmax(dim=-1)
        self.e5_vec_softmax = self.e5_lay_softmax(self.e5_vec_qkt_dk_12_head)
        # Context Vectors
        self.e5_vec_qktv = torch.matmul(self.e5_vec_softmax, self.e5_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e5__new_context_vecs_shape = self.e5_vec_qktv.shape[:-2] + (768,)
        self.e5_vec_qktv = self.e5_vec_qktv.view(*self.__e5__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e5_lay_selfout_dense = bert_model.encoder.layer[4].attention.output.dense
        self.e5_vec_selfout_dense = self.e5_lay_selfout_dense(self.e5_vec_qktv)
        # Add
        self.e5_vec_selfout_add = self.e5_vec_selfout_dense + self.e5_input_vecs
        # LayerNorm
        self.e5_lay_selfout_norm = bert_model.encoder.layer[4].attention.output.LayerNorm
        self.e5_vec_selfout_norm = self.e5_lay_selfout_norm(self.e5_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e5_lay_interm_dense = bert_model.encoder.layer[4].intermediate.dense
        self.e5_vec_interm_dense = self.e5_lay_interm_dense(self.e5_vec_selfout_norm)
        # Activation
        self.e5_vec_interm_gelu = gelu(self.e5_vec_interm_dense)
        # BertOutput
        # Dense
        self.e5_lay_out_dense = bert_model.encoder.layer[4].output.dense
        self.e5_vec_out_dense = self.e5_lay_out_dense(self.e5_vec_interm_gelu)
        # Add
        self.e5_vec_out_add = self.e5_vec_out_dense + self.e5_vec_selfout_norm
        # LayerNorm
        self.e5_lay_out_norm = bert_model.encoder.layer[4].output.LayerNorm
        self.e5_vec_out_norm = self.e5_lay_out_norm(self.e5_vec_out_add)

        self.e5_vec_layer_output = self.e5_vec_out_norm

    #Layer 6
        self.e6_input_vecs = self.e5_vec_layer_output
        self.e6_lay_div_transpose = bert_model.encoder.layer[5].attention.self.transpose_for_scores
        self.e6_vec_queries_12_head = bert_model.encoder.layer[5].attention.self.query(self.e6_input_vecs)
        self.e6_vec_queries_12_head_div = self.e6_lay_div_transpose(self.e6_vec_queries_12_head)
        # K
        self.e6_vec_keys_12_head = bert_model.encoder.layer[5].attention.self.key(self.e6_input_vecs)
        self.e6_vec_keys_12_head_div = self.e6_lay_div_transpose(self.e6_vec_keys_12_head)
        # V
        self.e6_vec_values_12_head = bert_model.encoder.layer[5].attention.self.value(self.e6_input_vecs)
        self.e6_vec_values_12_head_div = self.e6_lay_div_transpose(self.e6_vec_values_12_head)
        # QK.T
        self.e6_vec_qkt_12_head = torch.matmul(self.e6_vec_queries_12_head_div,self.e6_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e6_vec_qkt_dk_12_head = self.e6_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e6_lay_softmax = nn.Softmax(dim=-1)
        self.e6_vec_softmax = self.e6_lay_softmax(self.e6_vec_qkt_dk_12_head)
        # Context Vectors
        self.e6_vec_qktv = torch.matmul(self.e6_vec_softmax, self.e6_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e6__new_context_vecs_shape = self.e6_vec_qktv.shape[:-2] + (768,)
        self.e6_vec_qktv = self.e6_vec_qktv.view(*self.__e6__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e6_lay_selfout_dense = bert_model.encoder.layer[5].attention.output.dense
        self.e6_vec_selfout_dense = self.e6_lay_selfout_dense(self.e6_vec_qktv)
        # Add
        self.e6_vec_selfout_add = self.e6_vec_selfout_dense + self.e6_input_vecs
        # LayerNorm
        self.e6_lay_selfout_norm = bert_model.encoder.layer[5].attention.output.LayerNorm
        self.e6_vec_selfout_norm = self.e6_lay_selfout_norm(self.e6_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e6_lay_interm_dense = bert_model.encoder.layer[5].intermediate.dense
        self.e6_vec_interm_dense = self.e6_lay_interm_dense(self.e6_vec_selfout_norm)
        # Activation
        self.e6_vec_interm_gelu = gelu(self.e6_vec_interm_dense)
        # BertOutput
        # Dense
        self.e6_lay_out_dense = bert_model.encoder.layer[5].output.dense
        self.e6_vec_out_dense = self.e6_lay_out_dense(self.e6_vec_interm_gelu)
        # Add
        self.e6_vec_out_add = self.e6_vec_out_dense + self.e6_vec_selfout_norm
        # LayerNorm
        self.e6_lay_out_norm = bert_model.encoder.layer[5].output.LayerNorm
        self.e6_vec_out_norm = self.e6_lay_out_norm(self.e6_vec_out_add)

        self.e6_vec_layer_output = self.e6_vec_out_norm
    #Layer 7
        self.e7_input_vecs = self.e6_vec_layer_output
        self.e7_lay_div_transpose = bert_model.encoder.layer[6].attention.self.transpose_for_scores
        self.e7_vec_queries_12_head = bert_model.encoder.layer[6].attention.self.query(self.e7_input_vecs)
        self.e7_vec_queries_12_head_div = self.e7_lay_div_transpose(self.e7_vec_queries_12_head)
        # K
        self.e7_vec_keys_12_head = bert_model.encoder.layer[6].attention.self.key(self.e7_input_vecs)
        self.e7_vec_keys_12_head_div = self.e7_lay_div_transpose(self.e7_vec_keys_12_head)
        # V
        self.e7_vec_values_12_head = bert_model.encoder.layer[6].attention.self.value(self.e7_input_vecs)
        self.e7_vec_values_12_head_div = self.e7_lay_div_transpose(self.e7_vec_values_12_head)
        # QK.T
        self.e7_vec_qkt_12_head = torch.matmul(self.e7_vec_queries_12_head_div,self.e7_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e7_vec_qkt_dk_12_head = self.e7_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e7_lay_softmax = nn.Softmax(dim=-1)
        self.e7_vec_softmax = self.e7_lay_softmax(self.e7_vec_qkt_dk_12_head)
        # Context Vectors
        self.e7_vec_qktv = torch.matmul(self.e7_vec_softmax, self.e7_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e7__new_context_vecs_shape = self.e7_vec_qktv.shape[:-2] + (768,)
        self.e7_vec_qktv = self.e7_vec_qktv.view(*self.__e7__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e7_lay_selfout_dense = bert_model.encoder.layer[6].attention.output.dense
        self.e7_vec_selfout_dense = self.e7_lay_selfout_dense(self.e7_vec_qktv)
        # Add
        self.e7_vec_selfout_add = self.e7_vec_selfout_dense + self.e7_input_vecs
        # LayerNorm
        self.e7_lay_selfout_norm = bert_model.encoder.layer[6].attention.output.LayerNorm
        self.e7_vec_selfout_norm = self.e7_lay_selfout_norm(self.e7_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e7_lay_interm_dense = bert_model.encoder.layer[6].intermediate.dense
        self.e7_vec_interm_dense = self.e7_lay_interm_dense(self.e7_vec_selfout_norm)
        # Activation
        self.e7_vec_interm_gelu = gelu(self.e7_vec_interm_dense)
        # BertOutput
        # Dense
        self.e7_lay_out_dense = bert_model.encoder.layer[6].output.dense
        self.e7_vec_out_dense = self.e7_lay_out_dense(self.e7_vec_interm_gelu)
        # Add
        self.e7_vec_out_add = self.e7_vec_out_dense + self.e7_vec_selfout_norm
        # LayerNorm
        self.e7_lay_out_norm = bert_model.encoder.layer[6].output.LayerNorm
        self.e7_vec_out_norm = self.e7_lay_out_norm(self.e7_vec_out_add)

        self.e7_vec_layer_output = self.e7_vec_out_norm
    #Layer 8
        self.e8_input_vecs = self.e7_vec_layer_output
        self.e8_lay_div_transpose = bert_model.encoder.layer[7].attention.self.transpose_for_scores
        self.e8_vec_queries_12_head = bert_model.encoder.layer[7].attention.self.query(self.e8_input_vecs)
        self.e8_vec_queries_12_head_div = self.e8_lay_div_transpose(self.e8_vec_queries_12_head)
        # K
        self.e8_vec_keys_12_head = bert_model.encoder.layer[7].attention.self.key(self.e8_input_vecs)
        self.e8_vec_keys_12_head_div = self.e8_lay_div_transpose(self.e8_vec_keys_12_head)
        # V
        self.e8_vec_values_12_head = bert_model.encoder.layer[7].attention.self.value(self.e8_input_vecs)
        self.e8_vec_values_12_head_div = self.e8_lay_div_transpose(self.e8_vec_values_12_head)
        # QK.T
        self.e8_vec_qkt_12_head = torch.matmul(self.e8_vec_queries_12_head_div,self.e8_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e8_vec_qkt_dk_12_head = self.e8_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e8_lay_softmax = nn.Softmax(dim=-1)
        self.e8_vec_softmax = self.e8_lay_softmax(self.e8_vec_qkt_dk_12_head)
        # Context Vectors
        self.e8_vec_qktv = torch.matmul(self.e8_vec_softmax, self.e8_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e8__new_context_vecs_shape = self.e8_vec_qktv.shape[:-2] + (768,)
        self.e8_vec_qktv = self.e8_vec_qktv.view(*self.__e8__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e8_lay_selfout_dense = bert_model.encoder.layer[7].attention.output.dense
        self.e8_vec_selfout_dense = self.e8_lay_selfout_dense(self.e8_vec_qktv)
        # Add
        self.e8_vec_selfout_add = self.e8_vec_selfout_dense + self.e8_input_vecs
        # LayerNorm
        self.e8_lay_selfout_norm = bert_model.encoder.layer[7].attention.output.LayerNorm
        self.e8_vec_selfout_norm = self.e8_lay_selfout_norm(self.e8_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e8_lay_interm_dense = bert_model.encoder.layer[7].intermediate.dense
        self.e8_vec_interm_dense = self.e8_lay_interm_dense(self.e8_vec_selfout_norm)
        # Activation
        self.e8_vec_interm_gelu = gelu(self.e8_vec_interm_dense)
        # BertOutput
        # Dense
        self.e8_lay_out_dense = bert_model.encoder.layer[7].output.dense
        self.e8_vec_out_dense = self.e8_lay_out_dense(self.e8_vec_interm_gelu)
        # Add
        self.e8_vec_out_add = self.e8_vec_out_dense + self.e8_vec_selfout_norm
        # LayerNorm
        self.e8_lay_out_norm = bert_model.encoder.layer[7].output.LayerNorm
        self.e8_vec_out_norm = self.e8_lay_out_norm(self.e8_vec_out_add)

        self.e8_vec_layer_output = self.e8_vec_out_norm
    #Layer 9
        self.e9_input_vecs = self.e8_vec_layer_output
        self.e9_lay_div_transpose = bert_model.encoder.layer[8].attention.self.transpose_for_scores
        self.e9_vec_queries_12_head = bert_model.encoder.layer[8].attention.self.query(self.e9_input_vecs)
        self.e9_vec_queries_12_head_div = self.e9_lay_div_transpose(self.e9_vec_queries_12_head)
        # K
        self.e9_vec_keys_12_head = bert_model.encoder.layer[8].attention.self.key(self.e9_input_vecs)
        self.e9_vec_keys_12_head_div = self.e9_lay_div_transpose(self.e9_vec_keys_12_head)
        # V
        self.e9_vec_values_12_head = bert_model.encoder.layer[8].attention.self.value(self.e9_input_vecs)
        self.e9_vec_values_12_head_div = self.e9_lay_div_transpose(self.e9_vec_values_12_head)
        # QK.T
        self.e9_vec_qkt_12_head = torch.matmul(self.e9_vec_queries_12_head_div,self.e9_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e9_vec_qkt_dk_12_head = self.e9_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e9_lay_softmax = nn.Softmax(dim=-1)
        self.e9_vec_softmax = self.e9_lay_softmax(self.e9_vec_qkt_dk_12_head)
        # Context Vectors
        self.e9_vec_qktv = torch.matmul(self.e9_vec_softmax, self.e9_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e9__new_context_vecs_shape = self.e9_vec_qktv.shape[:-2] + (768,)
        self.e9_vec_qktv = self.e9_vec_qktv.view(*self.__e9__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e9_lay_selfout_dense = bert_model.encoder.layer[8].attention.output.dense
        self.e9_vec_selfout_dense = self.e9_lay_selfout_dense(self.e9_vec_qktv)
        # Add
        self.e9_vec_selfout_add = self.e9_vec_selfout_dense + self.e9_input_vecs
        # LayerNorm
        self.e9_lay_selfout_norm = bert_model.encoder.layer[8].attention.output.LayerNorm
        self.e9_vec_selfout_norm = self.e9_lay_selfout_norm(self.e9_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e9_lay_interm_dense = bert_model.encoder.layer[8].intermediate.dense
        self.e9_vec_interm_dense = self.e9_lay_interm_dense(self.e9_vec_selfout_norm)
        # Activation
        self.e9_vec_interm_gelu = gelu(self.e9_vec_interm_dense)
        # BertOutput
        # Dense
        self.e9_lay_out_dense = bert_model.encoder.layer[8].output.dense
        self.e9_vec_out_dense = self.e9_lay_out_dense(self.e9_vec_interm_gelu)
        # Add
        self.e9_vec_out_add = self.e9_vec_out_dense + self.e9_vec_selfout_norm
        # LayerNorm
        self.e9_lay_out_norm = bert_model.encoder.layer[8].output.LayerNorm
        self.e9_vec_out_norm = self.e9_lay_out_norm(self.e9_vec_out_add)

        self.e9_vec_layer_output = self.e9_vec_out_norm
    #Layer 10
        self.e10_input_vecs = self.e9_vec_layer_output
        self.e10_lay_div_transpose = bert_model.encoder.layer[9].attention.self.transpose_for_scores
        self.e10_vec_queries_12_head = bert_model.encoder.layer[9].attention.self.query(self.e10_input_vecs)
        self.e10_vec_queries_12_head_div = self.e10_lay_div_transpose(self.e10_vec_queries_12_head)
        # K
        self.e10_vec_keys_12_head = bert_model.encoder.layer[9].attention.self.key(self.e10_input_vecs)
        self.e10_vec_keys_12_head_div = self.e10_lay_div_transpose(self.e10_vec_keys_12_head)
        # V
        self.e10_vec_values_12_head = bert_model.encoder.layer[9].attention.self.value(self.e10_input_vecs)
        self.e10_vec_values_12_head_div = self.e10_lay_div_transpose(self.e10_vec_values_12_head)
        # QK.T
        self.e10_vec_qkt_12_head = torch.matmul(self.e10_vec_queries_12_head_div,self.e10_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e10_vec_qkt_dk_12_head = self.e10_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e10_lay_softmax = nn.Softmax(dim=-1)
        self.e10_vec_softmax = self.e10_lay_softmax(self.e10_vec_qkt_dk_12_head)
        # Context Vectors
        self.e10_vec_qktv = torch.matmul(self.e10_vec_softmax, self.e10_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e10__new_context_vecs_shape = self.e10_vec_qktv.shape[:-2] + (768,)
        self.e10_vec_qktv = self.e10_vec_qktv.view(*self.__e10__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e10_lay_selfout_dense = bert_model.encoder.layer[9].attention.output.dense
        self.e10_vec_selfout_dense = self.e10_lay_selfout_dense(self.e10_vec_qktv)
        # Add
        self.e10_vec_selfout_add = self.e10_vec_selfout_dense + self.e10_input_vecs
        # LayerNorm
        self.e10_lay_selfout_norm = bert_model.encoder.layer[9].attention.output.LayerNorm
        self.e10_vec_selfout_norm = self.e10_lay_selfout_norm(self.e10_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e10_lay_interm_dense = bert_model.encoder.layer[9].intermediate.dense
        self.e10_vec_interm_dense = self.e10_lay_interm_dense(self.e10_vec_selfout_norm)
        # Activation
        self.e10_vec_interm_gelu = gelu(self.e10_vec_interm_dense)
        # BertOutput
        # Dense
        self.e10_lay_out_dense = bert_model.encoder.layer[9].output.dense
        self.e10_vec_out_dense = self.e10_lay_out_dense(self.e10_vec_interm_gelu)
        # Add
        self.e10_vec_out_add = self.e10_vec_out_dense + self.e10_vec_selfout_norm
        # LayerNorm
        self.e10_lay_out_norm = bert_model.encoder.layer[9].output.LayerNorm
        self.e10_vec_out_norm = self.e10_lay_out_norm(self.e10_vec_out_add)

        self.e10_vec_layer_output = self.e10_vec_out_norm
    #Layer 11
        self.e11_input_vecs = self.e10_vec_layer_output
        self.e11_lay_div_transpose = bert_model.encoder.layer[10].attention.self.transpose_for_scores
        self.e11_vec_queries_12_head = bert_model.encoder.layer[10].attention.self.query(self.e11_input_vecs)
        self.e11_vec_queries_12_head_div = self.e11_lay_div_transpose(self.e11_vec_queries_12_head)
        # K
        self.e11_vec_keys_12_head = bert_model.encoder.layer[10].attention.self.key(self.e11_input_vecs)
        self.e11_vec_keys_12_head_div = self.e11_lay_div_transpose(self.e11_vec_keys_12_head)
        # V
        self.e11_vec_values_12_head = bert_model.encoder.layer[10].attention.self.value(self.e11_input_vecs)
        self.e11_vec_values_12_head_div = self.e11_lay_div_transpose(self.e11_vec_values_12_head)
        # QK.T
        self.e11_vec_qkt_12_head = torch.matmul(self.e11_vec_queries_12_head_div,self.e11_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e11_vec_qkt_dk_12_head = self.e11_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e11_lay_softmax = nn.Softmax(dim=-1)
        self.e11_vec_softmax = self.e11_lay_softmax(self.e11_vec_qkt_dk_12_head)
        # Context Vectors
        self.e11_vec_qktv = torch.matmul(self.e11_vec_softmax, self.e11_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e11__new_context_vecs_shape = self.e11_vec_qktv.shape[:-2] + (768,)
        self.e11_vec_qktv = self.e11_vec_qktv.view(*self.__e11__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e11_lay_selfout_dense = bert_model.encoder.layer[10].attention.output.dense
        self.e11_vec_selfout_dense = self.e11_lay_selfout_dense(self.e11_vec_qktv)
        # Add
        self.e11_vec_selfout_add = self.e11_vec_selfout_dense + self.e11_input_vecs
        # LayerNorm
        self.e11_lay_selfout_norm = bert_model.encoder.layer[10].attention.output.LayerNorm
        self.e11_vec_selfout_norm = self.e11_lay_selfout_norm(self.e11_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e11_lay_interm_dense = bert_model.encoder.layer[10].intermediate.dense
        self.e11_vec_interm_dense = self.e11_lay_interm_dense(self.e11_vec_selfout_norm)
        # Activation
        self.e11_vec_interm_gelu = gelu(self.e11_vec_interm_dense)
        # BertOutput
        # Dense
        self.e11_lay_out_dense = bert_model.encoder.layer[10].output.dense
        self.e11_vec_out_dense = self.e11_lay_out_dense(self.e11_vec_interm_gelu)
        # Add
        self.e11_vec_out_add = self.e11_vec_out_dense + self.e11_vec_selfout_norm
        # LayerNorm
        self.e11_lay_out_norm = bert_model.encoder.layer[10].output.LayerNorm
        self.e11_vec_out_norm = self.e11_lay_out_norm(self.e11_vec_out_add)

        self.e11_vec_layer_output = self.e11_vec_out_norm
        #Layer 12
        self.e12_input_vecs = self.e11_vec_layer_output
        self.e12_lay_div_transpose = bert_model.encoder.layer[11].attention.self.transpose_for_scores
        self.e12_vec_queries_12_head = bert_model.encoder.layer[11].attention.self.query(self.e12_input_vecs)
        self.e12_vec_queries_12_head_div = self.e12_lay_div_transpose(self.e12_vec_queries_12_head)
        # K
        self.e12_vec_keys_12_head = bert_model.encoder.layer[11].attention.self.key(self.e12_input_vecs)
        self.e12_vec_keys_12_head_div = self.e12_lay_div_transpose(self.e12_vec_keys_12_head)
        # V
        self.e12_vec_values_12_head = bert_model.encoder.layer[11].attention.self.value(self.e12_input_vecs)
        self.e12_vec_values_12_head_div = self.e12_lay_div_transpose(self.e12_vec_values_12_head)
        # QK.T
        self.e12_vec_qkt_12_head = torch.matmul(self.e12_vec_queries_12_head_div,self.e12_vec_keys_12_head_div.transpose(-1,-2))
        # QK.T / sqrt(d_k)
        self.e12_vec_qkt_dk_12_head = self.e12_vec_qkt_12_head / math.sqrt(64)
        # Attention矩阵
        self.e12_lay_softmax = nn.Softmax(dim=-1)
        self.e12_vec_softmax = self.e12_lay_softmax(self.e12_vec_qkt_dk_12_head)
        # Context Vectors
        self.e12_vec_qktv = torch.matmul(self.e12_vec_softmax, self.e12_vec_values_12_head_div).permute(0,2,1,3).contiguous()
        self.__e12__new_context_vecs_shape = self.e12_vec_qktv.shape[:-2] + (768,)
        self.e12_vec_qktv = self.e12_vec_qktv.view(*self.__e12__new_context_vecs_shape)
        # BertSelfOutput
        # Dense
        self.e12_lay_selfout_dense = bert_model.encoder.layer[11].attention.output.dense
        self.e12_vec_selfout_dense = self.e12_lay_selfout_dense(self.e12_vec_qktv)
        # Add
        self.e12_vec_selfout_add = self.e12_vec_selfout_dense + self.e12_input_vecs
        # LayerNorm
        self.e12_lay_selfout_norm = bert_model.encoder.layer[11].attention.output.LayerNorm
        self.e12_vec_selfout_norm = self.e12_lay_selfout_norm(self.e12_vec_selfout_add)
        # BertIntermediate
        # Dense
        self.e12_lay_interm_dense = bert_model.encoder.layer[11].intermediate.dense
        self.e12_vec_interm_dense = self.e12_lay_interm_dense(self.e12_vec_selfout_norm)
        # Activation
        self.e12_vec_interm_gelu = gelu(self.e12_vec_interm_dense)
        # BertOutput
        # Dense
        self.e12_lay_out_dense = bert_model.encoder.layer[11].output.dense
        self.e12_vec_out_dense = self.e12_lay_out_dense(self.e12_vec_interm_gelu)
        # Add
        self.e12_vec_out_add = self.e12_vec_out_dense + self.e12_vec_selfout_norm
        # LayerNorm
        self.e12_lay_out_norm = bert_model.encoder.layer[11].output.LayerNorm
        self.e12_vec_out_norm = self.e12_lay_out_norm(self.e12_vec_out_add)

        self.e12_vec_layer_output = self.e12_vec_out_norm
