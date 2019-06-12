from lib.PeterNN import DNN
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = [-5, -4, -3, -2, 0, 2, 3, 4, 5]
    y = [25, 16, 9, 4, 0, 4, 9, 16, 25]
    hidden_layer_specs = list()
    hidden_layer_specs.append({'layer': 'dense',
                               'units': 32,
                               'activation': DNN.Activation.tanh})
    hidden_layer_specs.append({'layer': 'dense',
                               'units': 32,
                               'activation': DNN.Activation.tanh})
    net = DNN(input_shape=(None, 1), output_size=1, hidden_layers=hidden_layer_specs)
    net.initiate()
    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))
    episodes = 200000
    print('Training...')
    for i in range(episodes):
        net.train_batch(x, y)
        if i % (episodes*0.05) == 0 and i != 0:
            print((i/episodes)*100, '%')

    test_values = np.arange(-5, 5, 0.2)
    test_values = np.reshape(test_values, (len(test_values), 1))
    prediction = net.predict(test_values)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(test_values, prediction)
    plt.show()