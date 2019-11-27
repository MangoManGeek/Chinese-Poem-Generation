from char2vec import Char2Vec
from char_dict import CharDict, end_of_sentence, start_of_sentence
from data_utils import batch_train_data
from paths import save_dir
from pron_dict import PronDict
from random import random
from singleton import Singleton
from utils import CHAR_VEC_DIM, NUM_OF_SENTENCES
import numpy as np
import os
import sys
import tensorflow as tf

_BATCH_SIZE = 64
_NUM_UNITS = 512

_model_path = os.path.join(save_dir, 'model')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GenerateModel(Singleton):
    def __init__(self):
        super(GenerateModel, self).__init__()

        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self.pron_dict = PronDict()

        # where to save checkpoint
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # self.key_encoder_GRU = tf.keras.layers.Bidirectional(
        #     tf.keras.layers.GRU(units=int(_NUM_UNITS / 2), return_sequences=True, return_state=True))
        # self.context_encoder_GRU = tf.keras.layers.Bidirectional(
        #     tf.keras.layers.GRU(units=int(_NUM_UNITS / 2), return_sequences=True, return_state=True))
        # self.attention = tfa.seq2seq.BahdanauAttention(units=_NUM_UNITS)
        # self.decoder_GRU = tf.keras.layers.GRUCell(units=_NUM_UNITS)
        # self.attention_wrapper = tfa.seq2seq.attention_wrapper(cell=self.decoder_GRU, attention_machanism=self.attention)
        # self.decoder = tfa.seq2seq.dynamic_decode()

        # self.dense_layer = tf.keras.layers.Dense(intput = )
        # self.dense = tf.keras.layers.Dense(input_dim=_NUM_UNITS, units=len(self.char_dict))
        self.encoder = Encoder()
        self.decoder = Decoder(len(self.char_dict))

    def generate(self, keywords):
        assert NUM_OF_SENTENCES == len(keywords)
        context = start_of_sentence()
        for keyword in keywords:
            keyword_data, keyword_length = self._fill_np_matrix(
                [keyword] * _BATCH_SIZE)
            context_data, context_length = self._fill_np_matrix(
                [context] * _BATCH_SIZE)

            keyword_state, context_output, final_output, final_state = self.encoder(keyword_data, context_data)
            char = start_of_sentence()
            for _ in range(7):
                decoder_input, decoder_input_length = \
                    self._fill_np_matrix([char])
                if char == start_of_sentence():
                    pass
                else:
                    keyword_state = final_state
                probs, final_state = self.decoder(keyword_state, context_output, decoder_input, decoder_input_length,
                                                  final_output, final_state)
                prob_list = self._gen_prob_list(probs, context)
                prob_sums = np.cumsum(prob_list)
                rand_val = prob_sums[-1] * random()
                for i, prob_sum in enumerate(prob_sums):
                    if rand_val < prob_sum:
                        char = self.char_dict.int2char(i)
                        break
                context += char
            context += end_of_sentence()
        return context[1:].split(end_of_sentence())

    def _gen_prob_list(self, probs, context):
        prob_list = probs.numpy().tolist()[0]
        prob_list[0] = 0
        prob_list[-1] = 0
        idx = len(context)
        used_chars = set(ch for ch in context)
        for i in range(1, len(prob_list) - 1):
            ch = self.char_dict.int2char(i)
            # Penalize used characters.
            if ch in used_chars:
                prob_list[i] *= 0.6
            # Penalize rhyming violations.
            if (idx == 15 or idx == 31) and not self.pron_dict.co_rhyme(ch, context[7]):
                prob_list[i] *= 0.2
            # Penalize tonal violations.
            if (idx > 2 and 2 == idx % 8) and not self.pron_dict.counter_tone(context[2], ch):
                prob_list[i] *= 0.4
            if (4 == idx % 8 or 6 == idx % 8) and not self.pron_dict.counter_tone(context[idx - 2], ch):
                prob_list[i] *= 0.4
        return prob_list

    # def _encoder(self, keyword_data, context_data):
    #     # keyword_length pending
    #     _, keyword_f_states, keyword_b_states = self.key_encoder_GRU(keyword_data)
    #     keyword_state = tf.concat([keyword_f_states, keyword_b_states], axis=1)
    #     context_output, _, _ = self.key_encoder_GRU(context_data)
    #     return keyword_state, context_output

    # def _decoder(self, keyword_state, context_output, decoder_input, decoder_input_length):
    #     attention = self.attention(context_output)
    #     a = 0
    #     attention_wrapper = tfa.seq2seq.attention_wrapper(cell=self.decoder_GRU(keyword_state), attention_machanism=attention, output_attention=False)
    #     final_output, final_state, _ = self.decoder(attention_wrapper)
    #     reshaped_outputs = self._reshape_decoder_outputs(final_output, decoder_input_length)
    #     logits = self.dense(reshaped_outputs)
    #     prob = tf.nn.softmax(logits)
    #     return prob, final_state
    #
    # def _reshape_decoder_outputs(self, decoder_outputs, decoder_input_length):
    #     """ Reshape decoder_outputs into shape [?, _NUM_UNITS]. """
    #     def concat_output_slices(idx, val):
    #         output_slice = tf.slice(
    #                 input=decoder_outputs,
    #                 begin=[idx, 0, 0],
    #                 size=[1, decoder_input_length[idx],  _NUM_UNITS])
    #         return tf.add(idx, 1),\
    #                 tf.concat([val, tf.squeeze(output_slice, axis=0)],
    #                         axis=0)
    #     tf_i = tf.constant(0)
    #     tf_v = tf.zeros(shape=[0, _NUM_UNITS], dtype=tf.float32)
    #     _, reshaped_outputs = tf.while_loop(
    #             cond=lambda i, v: i < _BATCH_SIZE,
    #             body=concat_output_slices,
    #             loop_vars=[tf_i, tf_v],
    #             shape_invariants=[tf.TensorShape([]),
    #                 tf.TensorShape([None, _NUM_UNITS])])
    #     tf.TensorShape([None, _NUM_UNITS]).\
    #             assert_same_rank(reshaped_outputs.shape)
    #     return reshaped_outputs

    def train(self, checkpoint, n_epochs):
        print("Training RNN-based generator ...")
        try:
            for epoch in range(n_epochs):
                batch_no = 0
                for keywords, contexts, sentences in batch_train_data(_BATCH_SIZE):
                    sys.stdout.write("[Seq2Seq Training] epoch = %d, line %d to %d ..." %
                                     (epoch, batch_no * _BATCH_SIZE,
                                      (batch_no + 1) * _BATCH_SIZE))
                    sys.stdout.flush()
                    self._train_a_batch(keywords, contexts, sentences)
                    batch_no += 1
                    if 0 == batch_no % 32:
                        checkpoint.save(file_prefix=save_dir)
                checkpoint.save(file_prefix=save_dir)
            print("Training is done.")
        except KeyboardInterrupt:
            print("Training is interrupted.")

    def _train_a_batch(self, keywords, contexts, sentences):
        keyword_data, keyword_length = self._fill_np_matrix(keywords)
        context_data, context_length = self._fill_np_matrix(contexts)
        decoder_input, decoder_input_length = self._fill_np_matrix(
            [start_of_sentence() + sentence[:-1] for sentence in sentences])
        targets = self._fill_targets(sentences)
        loss = 0
        with tf.GradientTape() as tape:
            keyword_state, context_output, final_output, final_state = self.encoder(keyword_data, context_data)
            probs, final_state = self.decoder(keyword_state, context_output, decoder_input, decoder_input_length,
                                              final_output, final_state)
            loss = loss_func()
        print(" loss =  %f" % loss)

    def loss_func(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def _fill_np_matrix(self, texts):
        max_time = max(map(len, texts))  # the len of keyword
        matrix = np.zeros([_BATCH_SIZE, max_time, CHAR_VEC_DIM],
                          dtype=np.float32)
        for i in range(_BATCH_SIZE):
            for j in range(max_time):
                matrix[i, j, :] = self.char2vec.get_vect(end_of_sentence())
        for i, text in enumerate(texts):
            matrix[i, : len(text)] = self.char2vec.get_vects(text)
        seq_length = [len(texts[i]) if i < len(texts) else 0 for i in range(_BATCH_SIZE)]
        return matrix, seq_length

    def _fill_targets(self, sentences):
        targets = []
        for sentence in sentences:
            targets.extend(map(self.char_dict.char2int, sentence))
        return targets


class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()

        self.key_encoder_GRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=int(_NUM_UNITS / 2), return_sequences=True, return_state=True))
        self.context_encoder_GRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=int(_NUM_UNITS / 2), return_sequences=True, return_state=True))

    def call(self, keyword_data, context_data):
        keyword_output, keyword_f_state, keyword_b_state = self.key_encoder_GRU(keyword_data)
        keyword_state = tf.concat([keyword_f_state, keyword_b_state], axis=1)
        context_bi_output, context_f_state, context_b_state = self.key_encoder_GRU(context_data)
        context_state = tf.concat([context_f_state, context_b_state], axis=1)
        final_output = tf.concat([keyword_output, context_bi_output], axis=1)
        final_state = tf.concat([keyword_state, context_state], axis=1)
        return keyword_state, context_bi_output, final_output, final_state


class BahdanauAttention(tf.keras.Model):

    def __init__(self):
        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(_NUM_UNITS)
        self.W2 = tf.keras.layers.Dense(_NUM_UNITS * 2)
        self.V = tf.keras.layers.Dense(1)

    def call(self, hidden_state, output):
        # hidden shape == (batch_size,hidden size)
        # hidden_with_time_axis shape == (batch_size,1,hidden size)
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)

        test = self.W1(output)
        test2 = self.W2(hidden_with_time_axis)

        # score shape == (batch_size,max_length,1)
        score = self.V(tf.nn.tanh(self.W1(output) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size,max_length,1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size,hidden_size)
        context_vector = attention_weights * output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


class Decoder(tf.keras.Model):
    def __init__(self, char_dict_len):
        super(Decoder, self).__init__()

        self.decoder_gru = tf.keras.layers.GRU(units=_NUM_UNITS, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(char_dict_len)

        self.attention = BahdanauAttention()

        # encoder_output shape = (batch_size,max_length,hidden_size)

    def call(self, keyword_state, context_output, decoder_input, decoder_input_length, final_output, final_state):
        context_vector = self.attention(final_state, final_output)

        # x shape after passing through embedding == (batch_size,1,embedding_dim)
        # x = self.embedding(x)

        # x shape after concatenation == (batch_size,1,embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), decoder_input], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.decoder_gru(x, initial_state=keyword_state)

        reshaped_outputs = self._reshape_decoder_outputs(output, decoder_input_length)

        logits = self.fc(reshaped_outputs)

        prob = tf.nn.softmax(logits)

        # output shape == (batch_size * 1,hidden_size)
        # output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size,vocab)
        # x = self.fc(output)

        return prob, state

        # final_output, final_state, _ = self.decoder(attention_wrapper)
        # reshaped_outputs = self._reshape_decoder_outputs(final_output, decoder_input_length)
        # logits = self.dense(reshaped_outputs)
        # prob = tf.nn.softmax(logits)
        # return prob, final_state

    def _reshape_decoder_outputs(self, decoder_outputs, decoder_input_length):
        """ Reshape decoder_outputs into shape [?, _NUM_UNITS]. """

        def concat_output_slices(idx, val):
            output_slice = tf.slice(
                input_=decoder_outputs,
                begin=[idx, 0, 0],
                size=[1, decoder_input_length[idx], _NUM_UNITS])
            return tf.add(idx, 1), \
                   tf.concat([val, tf.squeeze(output_slice, axis=0)],
                             axis=0)

        tf_i = tf.constant(0)
        tf_v = tf.zeros(shape=[0, _NUM_UNITS], dtype=tf.float32)
        _, reshaped_outputs = tf.while_loop(
            cond=lambda i, v: i < _BATCH_SIZE,
            body=concat_output_slices,
            loop_vars=[tf_i, tf_v],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, _NUM_UNITS])])
        tf.TensorShape([None, _NUM_UNITS]). \
            assert_same_rank(reshaped_outputs.shape)
        return reshaped_outputs


if __name__ == '__main__':
    a = 0
    generator = GenerateModel()
    keywords = ['四时', '变', '雪', '新']
    poem = generator.generate(keywords)
    for sentence in poem:
        print(sentence)
