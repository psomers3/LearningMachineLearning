import tensorflow as tf


class DNN:
    class Activation:
        relu = 'relu'
        tanh = 'tanh'
        sigmoid = 'sigmoid'
        softmax = 'softmax'

    _activation_dict = {'relu': tf.nn.relu,
                        'tanh': tf.nn.tanh,
                        'sigmoid': tf.nn.sigmoid,
                        'softmax': tf.nn.softmax}

    _layer_type_dict = {'dense': tf.layers.dense}

    def __init__(self, input_shape, output_size, hidden_layers=None, graph=None, session=None):
        # tensorflow graph that this network will belong to
        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph
        with self.graph.as_default():
            if session is None:
                # give network its own tensorflow session to run on
                self.sess = tf.Session(graph=self.graph)
            else:
                self.sess = session

            # ************** Build actual Neural Net Structure ***************
            # ****************************************************************
            # The input layer that will convert all inputs to dtype
            self.state_in = tf.placeholder(shape=input_shape, dtype=tf.float64)
            self.true = tf.placeholder(shape=(None, output_size), dtype=tf.float64)

            # Add any requested hidden layers
            # TODO: Make check for validity of hidden_layers input more robust
            if hidden_layers is not False:
                # A temporary variable that will represent the highest layer of the network as it is built
                net = self.make_hidden_layers(hidden_layers)
            else:
                # A temporary variable that will represent the highest layer of the network as it is built
                net = self.state_in

            # The output layer that will be returned
            self.output = tf.layers.dense(net, units=output_size)
            # ************ END Neural Net Construction **********************
            # ***************************************************************

            # A Tensor to calculate the index of the highest output value
            self.chosen_action = tf.argmax(self.output, 1)

            # Determine the loss function to be used.
            # might be better to make a method to be able to change this externally
            self.loss = tf.losses.mean_squared_error(labels=self.true, predictions=self.output)
            # Set the optimizer
            # same comment as for the loss function
            self.optimizer = tf.train.AdagradOptimizer(0.01)
            self.train = self.optimizer.minimize(self.loss)

    def initiate(self):
        with self.graph.as_default():
            # initialize all global variables for the session
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def make_hidden_layers(self, params):
        with self.graph.as_default():
            # Grab input layer for reference
            net = self.state_in
            for layer in params:
                layer_params = dict()
                if layer.get('activation') is not None:
                    layer_params['activation'] = self._activation_dict[layer['activation']]

                if layer.get('units') is not None:
                    layer_params['units'] = layer['units']
                else:
                    # TODO: Raise an error and inform user
                    break

                if layer.get('layer') is not None:
                    net = self._layer_type_dict[layer['layer']](net, **layer_params)
                else:
                    # TODO: Raise an error and inform user
                    break
            return net

    def train_batch(self, x, y):
        feed_dict = {self.state_in: x, self.true: y}
        _, loss = self.sess.run((self.train, self.loss), feed_dict=feed_dict)

    def predict(self, x):
        return self.sess.run(self.output, feed_dict={self.state_in: x})