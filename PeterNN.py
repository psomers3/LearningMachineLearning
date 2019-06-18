from lib.PeterNN import DNN
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = [-4, -3, -2, 0, 2, 3, 4]
    y = [16, 9, 4, 0, 4, 9, 16]
    hidden_layer_specs = list()
    hidden_layer_specs.append({'layer': 'dense',
                               'units': 32,
                               'activation': 'relu'})
    hidden_layer_specs.append({'layer': 'dense',
                               'units': 32,
                               'activation': 'relu'})
    hidden_layer_specs.append({'layer': 'dense',
                               'units': 1})
    net = DNN(input_shape=(None, 1), layers=hidden_layer_specs)
    net.initiate()

    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))
    episodes = 100000
    print('Training...')
    for i in range(episodes):
        net.train_batch(x, y)
        if i % (episodes*0.05) == 0 and i != 0:
            print((i/episodes)*100, '%')

    json = net.write_model_to_json('./test.json')

    test_values = np.arange(-5, 5, 0.2)
    test_values = np.reshape(test_values, (len(test_values), 1))
    prediction_1 = net.predict(test_values)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(test_values, prediction_1)

    plt.show()