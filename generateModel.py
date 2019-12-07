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


class GenerateModel(tf.keras.Model):
    def __init__(self, istrain):
        super(GenerateModel, self).__init__()

        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self.learning_rate = 0.01

        self.encoder = Encoder()
        self.decoder = Decoder(len(self.char_dict), istrain)

        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder, decoer=self.decoder, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, save_dir, max_to_keep=3)

    def generate(self, keywords):
        if not tf.train.get_checkpoint_state(save_dir):
            print("Please train the model first! (./train.py -g)")
            sys.exit(1)
        # try:
        #     self.checkpoint.restore(self.manager.latest_checkpoint).assert_consumed()
        #     print("Checkpoint is loaded successfully !")
        # except AssertionError:
        #     print("Fail to load checkpoint. Please train the model first! (./train.py -g)")
        self.checkpoint.restore(self.manager.latest_checkpoint)
        print("Checkpoint is loaded successfully !")
        assert NUM_OF_SENTENCES == len(keywords)
        context = start_of_sentence()
        pron_dict = PronDict()
        for keyword in keywords:
            keyword_data, keyword_length = self._fill_np_matrix(
                [keyword] * _BATCH_SIZE)
            context_data, context_length = self._fill_np_matrix(
                [context] * _BATCH_SIZE)

            keyword_state, context_output, final_output, final_state, context_state = self.encoder(keyword_data,
                                                                                                   context_data)
            char = start_of_sentence()
            for _ in range(7):
                decoder_input, decoder_input_length = \
                    self._fill_np_matrix([char])
                if char == start_of_sentence():
                    pass
                else:
                    keyword_state = final_state
                probs, final_state, _ = self.decoder(keyword_state, context_output, decoder_input,
                                                          decoder_input_length, final_output, final_state,
                                                          context_state)
                prob_list = self._gen_prob_list(probs, context, pron_dict)
                prob_sums = np.cumsum(prob_list)
                rand_val = prob_sums[-1] * random()
                for i, prob_sum in enumerate(prob_sums):
                    if rand_val < prob_sum:
                        char = self.char_dict.int2char(i)
                        break
                context += char
            context += end_of_sentence()
        return context[1:].split(end_of_sentence())

    def _gen_prob_list(self, probs, context, pron_dict):
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
            if (idx == 15 or idx == 31) and \
                    not pron_dict.co_rhyme(ch, context[7]):
                prob_list[i] *= 0.2
            # Penalize tonal violations.
            if idx > 2 and 2 == idx % 8 and \
                    not pron_dict.counter_tone(context[2], ch):
                prob_list[i] *= 0.4
            if (4 == idx % 8 or 6 == idx % 8) and \
                    not pron_dict.counter_tone(context[idx - 2], ch):
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

    def train(self, n_epochs):
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
                        self.manager.save()
                self.manager.save()
            print("Training is done.")
        except KeyboardInterrupt:
            print("Training is interrupted.")

    def _train_a_batch(self, keywords, contexts, sentences):
        keyword_data, keyword_length = self._fill_np_matrix(keywords)
        context_data, context_length = self._fill_np_matrix(contexts)
        decoder_input, decoder_input_length = self._fill_np_matrix(
            [start_of_sentence() + sentence[:-1] for sentence in sentences])
        targets = self._fill_targets(sentences)

        # sentences is from data_utils -->
        # 澄潭皎镜石崔巍$ 石   ^
        # 万壑千岩暗绿苔$	暗	^澄潭皎镜石崔巍$

        # loss, learning_rate = 0
        with tf.GradientTape() as tape:
            keyword_state, context_output, final_output, final_state, context_state = self.encoder(keyword_data,
                                                                                                   context_data)
            probs, final_state, logits = self.decoder(keyword_state, context_output, decoder_input,
                                                      decoder_input_length,
                                                      final_output, final_state, context_state)
            loss = self.loss_func(targets, logits)  # self???
            print(" loss =  %f" % loss)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # return batch_loss #batch_loss??

    def loss_func(self, targets, logits):
        labels = tf.one_hot(targets, depth=len(self.char_dict))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(loss)

    def learning_rate_func(self, loss):
        learning_rate = tf.clip_by_value(tf.multiply(1.6e-5, tf.pow(2.1, loss)), clip_value_min=0.0002,
                                         clip_value_max=0.02)
        return learning_rate

    def _fill_targets(self, sentences):
        targets = []
        for sentence in sentences:
            targets.extend(map(self.char_dict.char2int, sentence))
        return targets

    def _fill_np_matrix(self, texts):
        max_time = max(map(len, texts))  # the len of keyword
        matrix = np.zeros([_BATCH_SIZE, max_time, CHAR_VEC_DIM],
                          dtype=np.float32)
        for i in range(_BATCH_SIZE):
            for j in range(max_time):
                matrix[i, j, :] = self.char2vec.get_vect(end_of_sentence())
        for i, text in enumerate(texts):
            matrix[i, : len(text)] = self.char2vec.get_vects(text)
        seq_length = [len(texts[i]) if i < len(texts) else 0 \
                      for i in range(_BATCH_SIZE)]
        return matrix, seq_length


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

        return keyword_state, context_bi_output, final_output, final_state, context_state


class BahdanauAttention(tf.keras.Model):

    def __init__(self):
        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(_NUM_UNITS)
        # self.W2 = tf.keras.layers.Dense(_NUM_UNITS*2)
        self.W2 = tf.keras.layers.Dense(_NUM_UNITS)

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
    def __init__(self, char_dict_len, isTrain):
        super(Decoder, self).__init__()

        self.train = isTrain

        self.decoder_gru = tf.keras.layers.GRU(units=_NUM_UNITS * 2, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(char_dict_len)

        self.attention = BahdanauAttention()

        # encoder_output shape = (batch_size,max_length,hidden_size)

    def call(self, keyword_state, context_output, decoder_input, decoder_input_length, final_output, final_state,
             context_state):
        context_vector = self.attention(final_state, final_output)
        test2 = np.repeat(context_vector[:, np.newaxis, :], 1, axis=1)
        if self.train:
            test2 = np.repeat(context_vector[:, np.newaxis, :], 8, axis=1)
        x_test = tf.concat([test2, decoder_input], axis=2)

        output, state = self.decoder_gru(x_test, initial_state=final_state)
        reshaped_outputs = self._reshape_decoder_outputs(output, decoder_input_length)
        logits = self.fc(reshaped_outputs)  # add bias??
        prob = tf.nn.softmax(logits)

        return prob, state, logits


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
