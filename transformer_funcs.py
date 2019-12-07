import numpy as np
import tensorflow as tf
import numpy as np


def Self_Attention(K, V, Q, use_mask=False):
    """
    STUDENT MUST WRITE:

    This functions runs a single attention head.

    :param K: is [batch_size x window_size_keys x embedding_size]
    :param V: is [batch_size x window_size_values x embedding_size]
    :param Q: is [batch_size x window_size_queries x embedding_size]
    :return: attention
    """

    window_size_queries = Q.get_shape()[1]  # window size of queries
    window_size_keys = K.get_shape()[1]  # window size of keys
    mask = tf.convert_to_tensor(
        value=np.transpose(np.tril(np.ones((window_size_queries, window_size_keys)) * np.NINF, -1), (1, 0)),
        dtype=tf.float32)
    atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

    # TODO:
    # 1) compute attention weights using queries and key matrices (if use_mask==True, then make sure to add the attention mask before softmax)
    # 2) build new embeddings by applying the attention weights to the values matrices

    # Check lecture slides for how to compute self-attention
    # Remember:
    # - Q is [batch_size x window_size_queries x embedding_size]
    # - V is [batch_size x window_size_values x embedding_size]
    # - K is [batch_size x window_size_keys x embedding_size]
    # - Mask is [batch_size x window_size_queries x window_size_keys]

    # Here, queries are matmuled with the transpose of keys to produce for every query vector, weights per key vector.
    # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
    # Those weights are then used to create linear combinations of the corresponding values for each query.
    # Those queries will become the new embeddings.

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

        # TODO:
        # Initialize the weight matrices for K, V, and Q.
        # They should be able to multiply an input_size vector to produce an output_size vector
        # Hint: use self.add_weight(...)

        self.K_weight = tf.Variable(tf.random.normal([input_size, output_size], stddev=.1))
        self.V_weight = tf.Variable(tf.random.normal([input_size, output_size], stddev=.1))
        self.Q_weight = tf.Variable(tf.random.normal([input_size, output_size], stddev=.1))

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
        """

        # TODO:
        # - Apply 3 matrices to turn inputs into keys, values, and queries. You will need to use tf.tensordot for this.
        # - Call self_attention with the keys, values, and queries, and with self.use_mask.

        K = tf.tensordot(inputs_for_keys, self.K_weight, axes=([2], [0]))
        V = tf.tensordot(inputs_for_values, self.V_weight, axes=([2], [0]))
        Q = tf.tensordot(inputs_for_queries, self.Q_weight, axes=([2], [0]))

        attention = Self_Attention(K, V, Q, self.use_mask)

        return attention


class Multi_Headed(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask):
        super(Multi_Headed, self).__init__()

        # TODO:
        # Initialize heads

        self.split_size = int(emb_sz / 3)
        self.head1 = Atten_Head(emb_sz, self.split_size, use_mask)
        self.head2 = Atten_Head(emb_sz, self.split_size, use_mask)
        self.head3 = Atten_Head(emb_sz, emb_sz - 2 * self.split_size, use_mask)

        self.dense_layer = tf.keras.layers.Dense(input_dim=emb_sz, units=emb_sz)

    # @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        FOR CS2470 STUDENTS:

        This functions runs a multiheaded attention layer.

        Requirements:
            - Splits data for 3 different heads of size embed_sz/3
            - Create three different attention heads
            - Concatenate the outputs of these heads together
            - Apply a linear layer

        :param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
        """

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

    # @tf.function
    def call(self, inputs):
        """
        This functions creates a feed forward network as described in 3.3
        https://arxiv.org/pdf/1706.03762.pdf

        Requirements:
        - Two linear layers with relu between them

        :param inputs: input tensor [batch_size x window_size x embedding_size]
        :return: tensor [batch_size x window_size x embedding_size]
        """
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

    # @tf.function
    def call(self, inputs, context=None):
        """
        This functions calls a transformer block.

        There are two possibilities for when this function is called.
            - if self.is_decoder == False, then:
                1) compute unmasked attention on the inputs
                2) residual connection and layer normalization
                3) feed forward layer
                4) residual connection and layer normalization

            - if self.is_decoder == True, then:
                1) compute MASKED attention on the inputs
                2) residual connection and layer normalization
                3) computed UNMASKED attention using context
                4) residual connection and layer normalization
                5) feed forward layer
                6) residual layer and layer normalization

        If the multi_headed==True, the model uses multiheaded attention (Only 2470 students must implement this)

        :param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
        :context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
            default=None, This is context from the encoder to be used as Keys and Values in self-attention function
        """

        atten_out = self.self_atten(inputs, inputs, inputs)
        atten_out += inputs
        atten_normalized = self.layer_norm(atten_out)

        if self.is_decoder:
            assert context is not None, "Decoder blocks require context"

#             context_atten_out = self.self_context_atten(atten_normalized, context, context)
            context_atten_out = self.self_context_atten(context, context, atten_normalized)
            context_atten_out += atten_normalized
            atten_normalized = self.layer_norm(context_atten_out)

        ff_out = self.ff_layer(atten_normalized)
        ff_out += atten_normalized
        ff_norm = self.layer_norm(ff_out)

        return tf.nn.relu(ff_norm)


class Position_Encoding_Layer(tf.keras.layers.Layer):
    # def __init__(self, window_sz, emb_sz):
    #     super(Position_Encoding_Layer, self).__init__()
    #     self.positional_embeddings = self.add_weight("pos_embed", shape=[window_sz, emb_sz], trainable=False)

    def __init__(self, window_sz, emb_sz):
        super(Position_Encoding_Layer, self).__init__()
        self.positional_embeddings = self.add_weight("pos_embed", shape=[window_sz, emb_sz], trainable=False)
        # self.num_units = num_units
        # self.masking = masking
        # self.scale = scale
        # self.scope = scope
        # self.reuse = reuse

    # @tf.function
    def call(self, x):
        """
        Adds positional embeddings to word embeddings.

        :param x: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
        :return: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
        """
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
