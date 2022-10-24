import numpy as np
import networkx as nx
import datetime
import matplotlib.pyplot as plt


class Layer:
    """
    Define a network of n neurons.

    Parameters

        neurons : int   The number of neurons in this neural layer.

        input : list    The input for the layer to transform. Determines the number of weights per neuron.

        bias : int      The max range of neuron biases.

    Attributes

        neurons : list  The neurons in the layer. Each is a Neuron class.

    Example

        In: `l1 = Layer(10, 4)`
        Out: `<__main__.Layer at 0x1142391c0>`

        In: `l1.neurons[0].weights`
        Out: `array([ 0.31, -0.55, -0.37,  0.98])`
    """

    def __init__(self, neurons: int, input: list, bias: int):

        self.neurons = []

        self.layer_outputs = []

        self.weights = []

        self.biases = []

        self.output = []

        for i in range(neurons):

            self.neurons.append(Neuron(input, bias))

        for idx, item in enumerate(self.neurons):

            self.weights.append(self.neurons[idx].weights)

        for idx, item in enumerate(self.neurons):

            self.biases.append(self.neurons[idx].bias)

        self.transform(input)

    def count(self):
        
        '''Return `Neuron` count.'''
        
        return len(self.neurons)

    def describe(self):

        '''Describe the `Neurons` in this `Layer`.'''

        for idx, item in enumerate(self.neurons):

            print(f'n{idx}: {list(item.weights)} + {item.bias}')

    def transform(self, input: list):
        '''Pass an input through the neural layer.
        
        Alternate pythonic efficient dot-product method:

        import operator
        sum(map(operator.mul, vector1, vector2))

        '''

        self.output = np.dot(self.weights, input) + self.biases
        

class Neuron:
    """
    Define a neuron, with n weights, and one bias.

    Weights are a numpy.random uniform float between `-1` and `1`, to two decimal places.
    Bias is a numpy.random uniform float between `0` and `1`, to two decimal places.

    Parameters

        input : int     The neuron's input to transform. This determines the number of weights generated.

        bias : int      The neuron's bias max range.

    Attributes

        weights : list  The weights in this neuron.

        bias : int      This neuron's bias.
    
    Example

        In: `n1 = Neuron(3)`
        Out: `<__main__.Neuron at 0x1144b2850>`

        In: `n1.weights`
        Out: `array([ 0.05, -0.59, -0.44])`
    """

    def __init__(self, input: list, bias: int):

        self.bias = round(np.random.uniform(0, bias), 1)
        
        self.output = 0

        self.weights = np.random.uniform(-1, 1, len(input))

        self.inputs = []

        for i, j in enumerate(self.weights):

            self.weights[i] = round(self.weights[i], 2)

        self.G = nx.DiGraph(creation_time=datetime.datetime.now())

        self.transform(input)

    def transform(self, input: list):

        self.inputs = input

        self.output = np.dot(self.weights, input) + self.bias

    def describe(self):

        print(f'n0: ( {self.inputs} * {list(self.weights)} ) + {self.bias} = {self.output}')

    def graph(self):

        pos= nx.circular_layout(self.G)

        nx.draw_networkx_nodes(self.G, pos, node_size=1000)

        nx.draw_networkx_edges(self.G, pos)

        nx.draw_networkx_labels(self.G, pos)

        edge_labels = nx.get_edge_attributes(self.G, 'weight')

        nx.draw_networkx_edge_labels(self.G, pos, edge_labels)
        
        ax = plt.gca()
        
        ax.margins(0.3)

        return ax
