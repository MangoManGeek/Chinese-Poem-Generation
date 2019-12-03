import tensorflow as tf
import numpy as np
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size

        self.vocab_size = vocab_size
        self.window_size = 5
        self.embedding_size = 300#_
        self.batch_size = 100#_

        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)

        self.gru_out_dimension=300

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 

        self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=.1))

        self.gru=tf.keras.layers.GRU(self.gru_out_dimension,return_sequences=True,return_state=True)
        self.dense=tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.
        
        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn
       
        -Note 1: If you use an LSTM, the final_state will be the last two outputs of calling the rnn.
        If you use a GRU, it will just be the second output.

        -Note 2: You only need to use the initial state during generation. During training and testing it can be None.
        """

        input_size=np.shape(inputs)[0]
        #print(type(inputs[0]))
        #print (np.shape(inputs))
        inputs=[np.array(a) for a in inputs]
        inputs=np.array(inputs)
        #print (type(inputs))
        #print (type(inputs[0]))
        #print (type(inputs[0][0]))
        #print (np.shape(inputs))
        inputs=tf.constant(inputs)
        embeddings=tf.nn.embedding_lookup(self.E, inputs)   #3D tensor, batch size * window size * embedding size

        gru_out, final_state=self.gru(embeddings,initial_state=initial_state)

        #dense_in=tf.reshape(gru_out,[-1,self.gru_out_dimension])
        dense_in=gru_out
        logits=self.dense(dense_in)

        prbs=tf.nn.softmax(logits)
       
        return prbs,final_state



    def loss(self, logits, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        prbs=logits
        labels=[np.array(a) for a in labels]
        labels=np.array(labels)
        #print (np.shape(labels))
        labels=tf.constant(labels)
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,prbs, from_logits=False))

        #return None

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.
    
    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """

    epoch_size=np.shape(train_labels)[0]
    batch_size=model.batch_size
    window_size=model.window_size

    inputs_windows=[]
    labels_windows=[]

    for i in range(0,epoch_size,window_size):
        inputs_windows.append(train_inputs[i:i+window_size])
        labels_windows.append(train_labels[i:i+window_size])
    inputs_windows=inputs_windows[:-1]
    labels_windows=labels_windows[:-1]

    inputs_windows=np.array(inputs_windows)
    labels_windows=np.array(labels_windows)
    #print (inputs_windows)

    numOfWindows=np.shape(inputs_windows)[0]

    
    for i in range(0,numOfWindows,batch_size):

        batch_inputs=inputs_windows[i:i+batch_size]
        batch_labels=labels_windows[i:i+batch_size]

        with tf.GradientTape() as tape:

            prbs, final_state=model.call(batch_inputs, None)
            loss=model.loss(prbs, batch_labels)


        gradients = tape.gradient(loss, model.trainable_variables)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % (batch_size*10) == 0:
            #train_acc = model.loss(model(batch_inputs,initial_state=None)[0], batch_labels)
            train_acc=loss
            print("\rLoss on training set after {} training steps: {}".format(i, 1.0*train_acc),end="")



def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples
    
    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set

    Note: perplexity is exp(total_loss/number of predictions)

    """
    """
    epoch_size=np.shape(test_labels)[0]
    window_size=model.window_size

    inputs_windows=[]
    labels_windows=[]

    for i in range(0,epoch_size,window_size):
        inputs_windows.append(test_inputs[i:i+window_size])
        labels_windows.append(test_labels[i:i+window_size])

    inputs_windows=np.asarray(inputs_windows)
    labels_windows=np.asarray(labels_windows)
    print(inputs_windows)
    """
    epoch_size=np.shape(test_labels)[0]
    window_size=model.window_size
    extra=epoch_size%window_size

    new_inputs=test_inputs[:len(test_inputs)-extra]
    new_labels=test_labels[:len(test_labels)-extra]

    inputs_windows=np.reshape(new_inputs,[-1,window_size])
    labels_windows=np.reshape(new_labels,[-1,window_size])




    prbs,final_state=model.call(inputs_windows,None)
    loss=model.loss(prbs,labels_windows)
    return np.exp(loss)
    

def generate_sentence(word1, length, vocab,model):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    This is only for your own exploration. What do the sequences your RNN generates look like?
    
    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits,previous_state = model.call(next_input,previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))



def main():
    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see very, very small perplexities.

    import time
    start_time=time.time()


    print ('Loading Dataset')

    train_file='data/train_rnn_plan_data.txt'
    test_file='data/test_rnn_plan_data.txt'

    vocab_dict, train_data, test_data=get_data(train_file, test_file)

    #train_data=np.array(train_data)
    #test_data=np.array(test_data)
   
    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs=train_data[:len(train_data)-1]
    train_labels=train_data[1:]

    test_inputs=test_data[:len(test_data)-1]
    test_labels=test_data[1:]

    # TODO: initialize model and tensorflow variables
    model = Model(len(vocab_dict))

    num_epochs=10
    print ('Training...')
    for i in range(num_epochs):
        # TODO: Set-up the training step
        print ('EPOCH '+str(i))
        train(model, train_inputs,train_labels)

        train_time=time.time()-start_time
        print('\nTraining finished in {m} min {s} seconds'.format(m=train_time//60,s=train_time%60))

    # TODO: Set up the testing steps
    perplexity=test(model, test_inputs,test_labels)

    test_time=time.time()-start_time
    print('\nTesting finished in {m} min {s} seconds'.format(m=test_time//60,s=test_time%60))

    # Print out perplexity 
    print ('Test Perplexity: {p}'.format(p=perplexity))

    
if __name__ == '__main__':
    main()
