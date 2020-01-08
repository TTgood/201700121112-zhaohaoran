#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import tensorflow as tf
import codecs
import sys
import jieba
from tools import fenci

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.reset_default_graph()
# #### 1.参数设置。

# In[2]:


# 读取checkpoint的路径。9000表示是训练程序在第9000步保存的checkpoint。
# CHECKPOINT_PATH = "train_check_xhjall/attention_ckpt-7200"
CHECKPOINT_PATH = "train_15w/attention_ckpt-14800"
# 模型参数。必须与训练时的模型参数保持一致。
HIDDEN_SIZE = 1024                          # LSTM的隐藏层规模。
DECODER_LAYERS = 2                          # 解码器中LSTM结构的层数。
SRC_VOCAB_SIZE = 20000                      # 源语言词汇表大小。
TRG_VOCAB_SIZE = 20000                       # 目标语言词汇表大小。
SHARE_EMB_AND_SOFTMAX = True                # 在Softmax层和词向量层之间共享参数。

# 词汇表文件
SRC_VOCAB = "samples_15w/vocab.txt"
TRG_VOCAB = "samples_15w/vocab.txt"

# 词汇表中<sos>和<eos>的ID。在解码过程中需要用<sos>作为第一步的输入，并将检查
# 是否是<eos>，因此需要知道这两个符号的ID。
SOS_ID = 1
EOS_ID = 2


# #### 2.定义NMT模型和解码步骤。

# 定义NMTModel类来描述模型。
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) 
           for _ in range(DECODER_LAYERS)])

        # 为源语言和目标语言分别定义词向量。   
        self.src_embedding = tf.get_variable(
            "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
               "weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [TRG_VOCAB_SIZE])

    def inference(self, src_input):
        # 虽然输入只有一个句子，但因为dynamic_rnn要求输入是batch的形式，因此这里
        # 将输入句子整理为大小为1的batch。
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # 使用bidirectional_dynamic_rnn构造编码器。这一步与训练时相同。
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size, 
                dtype=tf.float32)
            # 将两个LSTM的输出拼接为一个张量。
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)    
        
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # 定义解码器使用的注意力机制。
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                HIDDEN_SIZE, enc_outputs,
                memory_sequence_length=src_size)

            # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络。
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=HIDDEN_SIZE)
   
        # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。
        MAX_DEC_LEN=100

        with tf.variable_scope("decoder/rnn/attention_wrapper", reuse=tf.AUTO_REUSE):
            # 使用一个变长的TensorArray来存储生成的句子。
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                dynamic_size=True, clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入。
            init_array = init_array.write(0, SOS_ID)
            # 调用attention_cell.zero_state构建初始的循环状态。循环状态包含
            # 循环神经网络的隐藏状态，保存生成句子的TensorArray，以及记录解码
            # 步数的一个整数step。
            init_loop_var = (
                attention_cell.zero_state(batch_size=1, dtype=tf.float32),
                init_array, 0)

            # tf.while_loop的循环条件：
            # 循环直到解码器输出<eos>，或者达到最大步数为止。
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量。
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                # 调用attention_cell向前计算一步。
                dec_outputs, next_state = attention_cell.call(
                    state=state, inputs=trg_emb)
                # 计算每个可能的输出单词对应的logit，并选取logit值最大的单词作为
                # 这一步的而输出。
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                # 线性映射，将循环网络输出映射成一个维度与词汇表大小相同的向量
                # softmax->logoits为加和为1的概率
                logits = (tf.matmul(output, self.softmax_weight)
                          + self.softmax_bias)
                # print(logits.shape)
                # 选出最可能的单词编号
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # print(logits[0][next_id])
                # 将这一步输出的单词写入循环状态的trg_ids中。
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            # 执行tf.while_loop，返回最终状态。
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


# #### 3.翻译一个测试句子。

# In[4]:
def chat(test_en_text, model):
    # 定义训练用的循环神经网络模型。
    # tf.reset_default_graph()


    # 定义个测试句子。
    # test_en_text = "And Japanese is good at assimilating new thing, direct with market demand link up with.<eos>"
    test_en_text.replace(",", "，")
    test_en_text.replace("？", "?")
    test_en_text = " ".join(fenci(test_en_text)).split() + ["<eos>"]

    # print(test_en_text)

    # 根据英文词汇表，将测试句子转为单词ID。
    with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
    src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                   for token in test_en_text]
    # print(test_en_ids)

    # 建立解码所需的计算图。
    output_op = model.inference(test_en_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_PATH)

    # 读取翻译结果。
    output_ids = sess.run(output_op)
    # print(output_ids)

    # 根据中文词汇表，将翻译结果转换为中文文字。
    with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    output_text = ''.join([trg_vocab[x] for x in output_ids])
    # output_text = output_text.replace("<eos>", "")
    # output_text = output_text.replace("<sos>", "")
    # # 输出翻译结果。
    # output_text = output_text.encode('utf8').decode(sys.stdout.encoding)
    result = output_text.encode('utf8').decode(sys.stdout.encoding)
    result_out = result[5:-5].replace("<unk>", "")
    if result_out == "":
        result_out = "我听不懂呢主人"

    sess.close()
    return result_out

# def main():
#     # 定义训练用的循环神经网络模型。
#     with tf.variable_scope("nmt_model", reuse=None):
#         model = NMTModel()
#     sess = tf.Session()
#     saver = tf.train.Saver()
#     # 定义个测试句子。
#     while True:
        
#         test_en_text = input("> ")
#         if test_en_text == "*":
#             break
#         print(test_en_text)
#         test_en_text = " ".join(jieba.cut(test_en_text)).split() + ["<eos>"]

    
#         # 根据英文词汇表，将测试句子转为单词ID。
#         with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
#             src_vocab = [w.strip() for w in f_vocab.readlines()]
#             src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
#         # test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
#         #                for token in test_en_text.split()]
#         test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
#                     for token in test_en_text]
#         # print(test_en_ids)

#         # 建立解码所需的计算图。
#         output_op = model.inference(test_en_ids)
    
#         saver.restore(sess, CHECKPOINT_PATH)

#         # 读取翻译结果。
#         output_ids = sess.run(output_op)
#         # print(output_ids)
    
#         # 根据中文词汇表，将翻译结果转换为中文文字。
#         with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
#             trg_vocab = [w.strip() for w in f_vocab.readlines()]
#         output_text = ''.join([trg_vocab[x] for x in output_ids])
    
#         # 输出翻译结果。
#         print(output_text.encode('utf8').decode(sys.stdout.encoding))
#     sess.close()
def make_model():
    with tf.variable_scope("nmt_model", reuse=tf.AUTO_REUSE):
        model = NMTModel()
    return model

def main():
    model = make_model()
    while True:
        test_in = input("> ")
        if test_in == "*":
            break
        out = chat(test_in, model)
        print(out)
        print()

if __name__ == "__main__":
    main()


# In[ ]:




