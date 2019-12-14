import numpy as np
import tensorflow as tf
import numpy as np


def Self_Attention(K, V, Q, use_mask=False):

    window_size_queries = Q.get_shape()[1]  # window size of queries
    window_size_keys = K.get_shape()[1]  # window size of keys
    mask = tf.convert_to_tensor(
        value=np.transpose(np.tril(np.ones((window_size_queries, window_size_keys)) * np.NINF, -1), (1, 0)),
        dtype=tf.float32)
    atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

    atten_weight = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(float(K.shape[2]))  # divide by d_k!!!!!

    if (use_mask):
        atten_weight = atten_weight + atten_mask

    softmax = tf.nn.softmax(atten_weight, -1)
    embedding = tf.matmul(softmax, V)

    return embedding


class Atten_Head(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, use_mask):
        super(Atten_Head, self).__init__()

        self.use_mask = use_mask

        self.K_weight = tf.Variable(tf.random.normal([input_size, output_size], stddev=.1))
        self.V_weight = tf.Variable(tf.random.normal([input_size, output_size], stddev=.1))
        self.Q_weight = tf.Variable(tf.random.normal([input_size, output_size], stddev=.1))

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

        K = tf.tensordot(inputs_for_keys, self.K_weight, axes=([2], [0]))
        V = tf.tensordot(inputs_for_values, self.V_weight, axes=([2], [0]))
        Q = tf.tensordot(inputs_for_queries, self.Q_weight, axes=([2], [0]))

        attention = Self_Attention(K, V, Q, self.use_mask)

        return attention


class Multi_Headed(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask):
        super(Multi_Headed, self).__init__()

        self.split_size = int(emb_sz / 3)
        self.head1 = Atten_Head(emb_sz, self.split_size, use_mask)
        self.head2 = Atten_Head(emb_sz, self.split_size, use_mask)
        self.head3 = Atten_Head(emb_sz, emb_sz - 2 * self.split_size, use_mask)

        self.dense_layer = tf.keras.layers.Dense(input_dim=emb_sz, units=emb_sz)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

        atten_head1 = self.head1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        atten_head2 = self.head2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        atten_head3 = self.head3(inputs_for_keys, inputs_for_values, inputs_for_queries)

        concate = tf.concat([atten_head1, atten_head2, atten_head3], 2)

        softmax = self.dense_layer(concate)

        return softmax


class Feed_Forwards(tf.keras.layers.Layer):
    def __init__(self, emb_sz):
        super(Feed_Forwards, self).__init__()

        self.layer_1 = tf.keras.layers.Dense(emb_sz, activation='relu')
        self.layer_2 = tf.keras.layers.Dense(emb_sz)

    @tf.function
    def call(self, inputs):
        layer_1_out = self.layer_1(inputs)
        layer_2_out = self.layer_2(layer_1_out)
        return layer_2_out


class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, emb_sz, is_decoder, multi_headed=False):
        super(Transformer_Block, self).__init__()

        self.ff_layer = Feed_Forwards(emb_sz)
        self.self_atten = Atten_Head(emb_sz, emb_sz, use_mask=is_decoder) if not multi_headed else Multi_Headed(emb_sz,
                                                                                                                use_mask=is_decoder)
        self.is_decoder = is_decoder
        if self.is_decoder:
            self.self_context_atten = Atten_Head(emb_sz, emb_sz, use_mask=False) if not multi_headed else Multi_Headed(
                emb_sz, use_mask=False)

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    @tf.function
    def call(self, inputs, context=None):

        atten_out = self.self_atten(inputs, inputs, inputs)
        atten_out += inputs
        atten_normalized = self.layer_norm(atten_out)

        if self.is_decoder:
            assert context is not None, "Decoder blocks require context"

            context_atten_out = self.self_context_atten(context, context, atten_normalized)
            context_atten_out += atten_normalized
            atten_normalized = self.layer_norm(context_atten_out)

        ff_out = self.ff_layer(atten_normalized)
        ff_out += atten_normalized
        ff_norm = self.layer_norm(ff_out)

        return tf.nn.relu(ff_norm)


class Position_Encoding_Layer(tf.keras.layers.Layer):
   
    def __init__(self, window_sz, emb_sz):
        super(Position_Encoding_Layer, self).__init__()
        self.positional_embeddings = self.add_weight("pos_embed", shape=[window_sz, emb_sz], trainable=False)

    @tf.function
    def call(self, x):
        
        return x + self.positional_embeddings

    # def call(self, x):
    #     '''Sinusoidal Positional_Encoding.
    #
    #     Args:
    #       inputs: A 2d Tensor with shape of (N, T).
    #       num_units: Output dimensionality
    #       zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    #       scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
    #       scope: Optional scope for `variable_scope`.
    #       reuse: Boolean, whether to reuse the weights of a previous layer
    #         by the same name.
    #
    #     Returns:
    #         A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    #     '''
    #
    #     E = x.get_shape().as_list()[-1]  # static
    #     N, T = tf.shape(x)[0], tf.shape(x)[1]  # dynamic
    #     with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
    #         # position indices
    #         position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)
    #
    #         # First part of the PE function: sin and cos argument
    #         position_enc = np.array([
    #             [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
    #             for pos in range(self.num_units)])
    #
    #         # Second part, apply the cosine to even columns and sin to odds.
    #         position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    #         position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    #         position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)
    #
    #         # lookup
    #         outputs = tf.nn.embedding_lookup(position_enc, position_ind)
    #
    #         # masks
    #         if self.masking:
    #             outputs = tf.where(tf.equal(x, 0), x, outputs)
    #
    #         return tf.to_float(outputs)
