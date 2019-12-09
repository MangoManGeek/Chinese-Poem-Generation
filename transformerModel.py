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
import transformer_funcs as transformer

_BATCH_SIZE = 64
_NUM_UNITS = 512
_WINDOW_SIZE = 7
_CURRENT_LEN = 1

_model_path = os.path.join(save_dir, 'model')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GenerateTransformerModel(tf.keras.Model):
    def __init__(self, isTrain):
        super(GenerateTransformerModel, self).__init__()

        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self.learning_rate = 0.001

        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # self.saver = tf.train.Saver(tf.global_variables())
        # self.trained = False

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

        self.encoder = Encoder(isTrain)
        self.decoder = Decoder(len(self.char_dict), isTrain)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

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

            encoder_output = self.encoder(keyword_data, context_data)
            char = start_of_sentence()
            for _ in range(7):
                decoder_input, decoder_input_length = \
                    self._fill_np_matrix([char])
                if char == start_of_sentence():
                    pass
                else:
                    # keyword_state = decoder_output
                    encoder_output = decoder_output
                probs, logits, decoder_output = self.decoder(encoder_output, decoder_input, decoder_input_length)
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

        #sentences is from data_utils --> (sentence, keyword, context)
        #澄潭皎镜石崔巍$ 石   ^
        #万壑千岩暗绿苔$	暗	^澄潭皎镜石崔巍$

        # loss, learning_rate = 0
        with tf.GradientTape() as tape:
            encoder_output = self.encoder(keyword_data, context_data)
            probs, logits, decoder_output = self.decoder(encoder_output, decoder_input, decoder_input_length)
            loss = self.loss_func(targets, logits, probs)
            # print(" loss =  %f" % loss)

            #where to initialize??
            learning_rate = self.learning_rate_func(loss)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            print(" loss =  %f, learning_rate = %f" % (loss, learning_rate))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        # return batch_loss #batch_loss??



    def loss_func(self, targets, logits, probs):
        labels = self.label_smoothing(tf.one_hot(targets, depth=len(self.char_dict)))
        # loss = tf.losses.softmax_cross_entropy(onehot_labels = labels, logits = logits)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False)
        return tf.reduce_mean(loss)

    def label_smoothing(self,inputs, epsilon=0.1):
        '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
        inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
        epsilon: Smoothing rate.

        For example,

        ```
        import tensorflow as tf
        inputs = tf.convert_to_tensor([[[0, 0, 1],
           [0, 1, 0],
           [1, 0, 0]],
          [[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0]]], tf.float32)

        outputs = label_smoothing(inputs)

        with tf.Session() as sess:
            print(sess.run([outputs]))

        >>
        [array([[[ 0.03333334,  0.03333334,  0.93333334],
            [ 0.03333334,  0.93333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334]],
           [[ 0.93333334,  0.03333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334],
            [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
        ```
        '''
        V = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / V)

    def learning_rate_func(self, loss):
        learning_rate = tf.clip_by_value(tf.multiply(1.6e-5, tf.pow(2.1, loss)), clip_value_min = 0.0002, clip_value_max = 0.02)
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

    def __init__(self, isTrain):
        super(Encoder, self).__init__()

        if isTrain:
            self.window_size = 25
        else:
            self.window_size = 9
        self.window_size = 25
        self.isTrain = isTrain
        self.dropout_rate = 0.1

        # Create positional encoder layers
        # self.pos_encoder_keyword = transformer.Position_Encoding_Layer(self.window_size, _NUM_UNITS)
        self.pos_encoder_context = transformer.Position_Encoding_Layer(self.window_size, _NUM_UNITS)

        #transformer encoder
        self.encoder = transformer.Transformer_Block(_NUM_UNITS, is_decoder=False, multi_headed=True)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, keyword_data, context_data):
        ####1. concate keyword and context first
        # keyword_pos = self.pos_encoder_keyword(keyword_data) #[64, 2, 512]
        # if self.isTrain:
        #     context_pos = self.pos_encoder_context(context_data)  # [64, 25/1, 512]
        # else:
        #     context_pos = context_data

        context_pos = context_data
        # if not self.isTrain:
        #     context_pos = self.pos_encoder_context(context_data) #[64, 25/1, 512]

        # context_pos = self.pos_encoder_context(context_data)  # [64, 25/1, 512]


        concate_pos_data = tf.concat([keyword_data, context_pos], axis=1)
        # concate_pos_data_drop = self.dropout(concate_pos_data)
        encoder_output = self.encoder(concate_pos_data)
        # if self.isTrain:
        #     encoder_output = self.dense_layer(encoder_output)

        return encoder_output


class Decoder(tf.keras.Model):
    def __init__(self, char_dict_len, isTrain):
        super(Decoder, self).__init__()

        if isTrain:
            self.window_size = 8
        else:
            self.window_size = 1

        self.isTrain = isTrain
        self.dropout_rate = 0.1

        # self.pos_decoder = transformer.Position_Encoding_Layer(self.window_size, _NUM_UNITS)

        self.decoder = transformer.Transformer_Block(_NUM_UNITS, is_decoder=True, multi_headed=True)

        self.dense_layer3 = tf.keras.layers.Dense(input_dim=_NUM_UNITS, units=char_dict_len)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # encoder_output shape = (batch_size,max_length,hidden_size)

    def call(self, encoder_output, decoder_input, decoder_input_length):
        # decoder_pos = self.pos_decoder(decoder_input)
        # decoder_pos = self.dropout(decoder_pos)
        decoder_pos = decoder_input
        if self.isTrain:
            lcm = np.lcm(self.window_size, encoder_output.shape[1])
            decoder_pos_reshape = np.repeat(decoder_pos, lcm/decoder_pos.shape[1], axis=1)
            encoder_reshape = np.repeat(encoder_output, lcm / encoder_output.shape[1], axis=1)
        else:
            decoder_pos_reshape = np.repeat(decoder_pos, encoder_output.shape[1], axis=1)
            encoder_reshape = encoder_output

        # decoder_pos_reshape = np.repeat(decoder_pos, encoder_output.shape[1], axis=1)
        # encoder_reshape = encoder_output

        # decoder_pos_reshape = self.dropout(decoder_pos_reshape)

        # decoder_pos_reshape = tf.broadcast_to(decoder_pos, shape=[64, encoder_output.shape[1], 512])

        decoder_output = self.decoder(decoder_pos_reshape, context=encoder_reshape)

        reshaped_outputs = self._reshape_decoder_outputs(decoder_output, decoder_input_length)

        logits = self.dense_layer3(reshaped_outputs)

        prob = tf.nn.softmax(logits, -1)

        # output shape == (batch_size * 1,hidden_size)
        # output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size,vocab)
        # x = self.fc(output)

        return prob, logits, decoder_output #or reshaped_output

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
    generator = GenerateTransformerModel(isTrain=False)
    keywords = ['四时', '变', '雪', '新']
    poem = generator.generate(keywords)
    for sentence in poem:
        print(sentence)
