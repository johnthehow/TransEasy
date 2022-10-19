# BERTEAT: BERT Explainability and Analysis Toolkit
## glassbox_bert.py
### Illustration of class variable groups
![glassbox_bert grouping](https://github.com/johnthehow/bertet/blob/master/figure1.png)
### Features
1. The glassbox_bert class returns an instance which encloses both data and model aftering being feed with a sentence
2. Unlike Huggingface BertModel instance, with glassbox_bert you can easily obtain all intermediate representations (45 in total) you can ever imagine, not just Attention and Hidden states.
3. The components in glassbox_bert are highly aligned with the concepts of Transformer encoder as in Vaswani et al. 2017.
4. All components (as class variables) are divied into six groups, namely:
	1. Symbolic: with class variable prefix ***sym_***:
	1. Pre-Embedding: with class variable prefix ***preemb_***:
	1. Multi-Head Self-Attention: with class variable prefix ***selfattn_***:
	1. Add & Norm 1: with class variable prefix ***addnorm1_***:
	1. Feedforward: with class variable prefix ***ffnn_***:
	1. Add & Norm 2: with class variable prefix ***addnorm2_***:
5. The results generated by BertModel Pipeline are stored with class variables with prefix ***pipeline_***:
6. The results generated by step-wise manual procedure are called with class variables with prefix ***manual_***:

### Example Usage
```python
from bertet import glassbox_bert
instance = glassbox_bert.glassbox_bert('an example sentence here')

```

### Note
1. In query, key and value weight matrices, every 64 rows correspond to a head
2. In query, key and value bias matrices, every 64 items correspond to a head
3. Matrix multiplications are done with XW.T, not other way round
3. The activation function of FFNN is gelu
4. The Softmax in Multi-Head Self-Attention is parameterized with axis=-1

### A complete list of class variables of glassbox_bert
* ****model****: a huggingface BertModel instance
* **tokenizer**: a huggingface BertTokenizer instance
* **pipeline_res**: pipeline result of BertModel
* **pipeline_attns**: pipeline attention of BertModel
* **pipeline_hiddens**: pipeline hidden states of BertModel
* **sym_sent**: plain text input sentence
* **sym_token_ids**: Bert vocabulary token ids (integer)
* **sym_tokens**: Word-Piece tokens by BertTokenizer (including [CLS] and [SEP])
* **sym_seq_len**: the length of word-piece-tokenized sentence (including [CLS] and [SEP])
* **preemb_vocab**: pre-trained static word embeddings (30522×768)
* **preemb_word_emb**: static word embeddings from preemb_vocab_emb
* **preemb_pos_emb**: position embeddings
* **preemb_seg_emb**: segmentation embeddings
* **preemb_sum_emb**: preemb_word_emb+preemb_pos_emb+preemb_seg_emb
* **preemb_norm_sum_emb**: LayerNorm(preemb_sum_emb)
* **selfattn_query_weight**: Query weight matrix 12×12×64×768
* **selfattn_key_weight**: Key weight matrix 12×12×64×768
* **selfattn_value_weight**: Value weight matrix 12×12×64×768
* **selfattn_query_bias**: Query bias vector 12×12×64
* **selfattn_key_bias**: Key bias vector 12×12×64
* **selfattn_value_bias**: Value bias vector 12×12×64
* **selfattn_query_hidden**: Query hidden states matrix 12×12×6×64
* **selfattn_key_hidden**: Key hidden states matrix 12×12×n×64
* **selfattn_value_hidden**: Value hidden states matrix 12×12×n×64
* **selfattn_qkt_hidden**: QK.T 12×12×n×n
* **selfattn_qkt_scale_hidden**: QK.T/sqrt(dk) 12×12×n×n
* **selfattn_qkt_scale_soft_hidden**: Attention:= Softmax(dim=-1)(selfattn_qkt_scale_hidden) 12×12×n×n
* **selfattn_attention**: ==selfattn_qkt_scale_soft_hidden 12×12×n×n
* **selfattn_contvec**: Attention×selfattn_value_hidden 12×12×n×64
* **selfattn_contvec_concat**: Concatenated selfattn_contvec 12×n×768
* **addnorm1_dense_weight**: Dense weight matrix of Add & Norm 1 layer 12×768×768
* **addnorm1_dense_bias**: Dense bias vector of Add & Norm 1 layer 12×768
* **addnorm1_dense_hidden**: Dense hidden states of Add & Norm 1 layer 12×n×768
* **addnorm1_add_hidden**: Residual connection: preemb_norm_sum_emb + addnorm1_dense_hidden 12×n×768
* **addnorm1_norm_hidden**: LayerNorm(addnorm1_add_hidden) 12×n×768
* **ffnn_dense_weight**: Dense weight matrix of Feed-forward layer 12×768×3072
* **ffnn_dense_bias**: Dense weight vector of Feed-forward layer 12×3072
* **ffnn_dense_hidden**: Dense hidden states of Feed-forward layer 12×n×3072
* **ffnn_dense_act**: ffnn_dense_hidden after gelu activation 12×n×3072
* **addnorm2_dense_weight**: Dense weight matrix of Add & Norm 2 layer 12×3072×768
* **addnorm2_dense_bias**: Dense bias vector of Add & Norm 2 layer 12×768
* **addnorm2_dense_hidden**: Dense hidden states of Add & Norm 2 layer 12×n×768
* **addnorm2_add_hidden**: Residual connection: addnorm1_norm_hidden + addnorm2_dense_hidden 12×n×768
* **addnorm2_norm_hidden**: LayerNorm(addnorm2_add_hidden) 12×n×768
* **manual_hiddens**: [preemb_norm_sum_emb, addnorm2_norm_hidden]

## bertanomy.py
### Features
1. Analysis of BERT hidden states, attention of words, sentences
1. Document is in the remarks of the script
1. Functions are divided into groups:
	1. Symbolic: with prefix **sym_**
		1. sym_better_tokenizer: a better tokenizer of BERT
		1. sym_bert_length_sents: select sentences with length determined by BERT tokenizer
	1. Pre-Embedding: with prefix **preemb_**
		1. preemb_word_preemb
		1. preemb_similar_preemb
	1. Attention: with prefix **attn_**
		1. attn_sent_attention_matrix
		1. attn_word_attention_row
	1. Hidden State: with prefix **hidden_**
		1. hidden_sent_hidden_states
		1. hidden_word_hidden_states_in_sent
		1. hidden_sent_hidden_vector
	1. Property: with prefix **prop_**
		1. prop_word_position_in_snet
		1. prop_word_attention_distance
		1. prop_word_most_attend_position
	1. Statistics: with prefix **stat_**
		1. stat_hidden_states_norm
		1. stat_word_hidden_norm
		1. stat_word_hidden_norm_in_sents
	1. Visualization: with prefix **viz_**
		1. viz_hist_word_hidden_states
		1. viz_barplot_attn_row
		1. viz_scatter_bert_preemb
	1. Pipeline: with prefix **pipe_**
		1. pipe_pipeline
