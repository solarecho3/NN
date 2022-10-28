import numpy as np
import networkx as nx
import datetime
import matplotlib.pyplot as plt
import pyvis


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

        self.G = pyvis.network.Network(height="700px")

        for i in range(neurons):

            self.neurons.append(Neuron(input, bias))

        for idx, item in enumerate(self.neurons):

            self.weights.append(self.neurons[idx].weights)

        for idx, item in enumerate(self.neurons):

            self.biases.append(self.neurons[idx].bias)

        self.transform(input)

    def count(self):

        """Return `Neuron` count."""

        return len(self.neurons)

    def describe(self):

        """Describe the `Neuron`s in this `Layer`."""

        for idx, item in enumerate(self.neurons):

            print(f"n{idx}: {list(item.weights)} + {item.bias}")

    def transform(self, input: list):

        """Pass an input through the neural layer."""

        self.output = np.dot(self.weights, input) + self.biases


class Neuron:
    """
    Define a `class Neuron`, with n weights, and one bias.

    Weights are a numpy.random uniform float between `-1` and `1`, to two decimal places.
    Bias is a numpy.random uniform float between `0` and `1`, to two decimal places.

    Parameters

        input : int     The neuron's input to transform. This determines the number of weights generated.

        bias : int      The neuron's bias max range.

    Attributes

        weights : list  The weights in this neuron.

        bias : int      This neuron's bias.

        output : float  This neuron's output after transform().

    Example

        In: `n1 = Neuron(3)`
        Out: `<__main__.Neuron at 0x1144b2850>`

        In: `n1.weights`
        Out: `array([ 0.05, -0.59, -0.44])`
    """

    def __init__(self, input: list, bias: int) -> np.array:

        self.bias = round(np.random.uniform(0, bias), 1)

        self.output = 0

        self.weights = np.random.uniform(-1, 1, len(input))

        self.inputs = input

        for i, j in enumerate(self.weights):

            self.weights[i] = round(self.weights[i], 2)

        self.G = pyvis.network.Network(height="700px")

        self.transform()

    def transform(self):

        self.output = np.dot(self.weights, self.inputs) + self.bias

    def describe(self):

        print(
            f"n0: ( {self.inputs} * {list(self.weights)} ) + {self.bias} = {self.output}"
        )

    def graph(self):

        """
        Displays a graph based on the VisJS library, ported by pyvis.

        Currently will graph improperly if given two identical inputs in the array. This would graph as a single node instead of two.
        """

        weights = ["w " + str(x) for x in self.weights]

        inputs = ["i " + str(x) for x in self.inputs]

        """
        This way works but the kwarg mass isn't recognized 
        self.G.add_nodes(weights, shape=['circle' for i in range(len(inputs))])

        self.G.add_nodes(inputs, shape=['box' for i in range(len(inputs))])

        """

        for i in weights:

            self.G.add_node(
                i, shape="circle", mass=2, level=2, title="weight", color="#ff4f78"
            )

        for i in inputs:

            self.G.add_node(
                i, shape="box", mass=10, level=1, title="input", color="#ffa74f"
            )

        self.G.add_node(
            "b " + str(self.bias),
            mass=self.bias * 150,
            shape="circle",
            level=3,
            title="bias",
            color="#4fffaa",
        )

        for i in weights:

            for j in inputs:

                self.G.add_edge(i, j)

                self.G.add_edge(i, "b " + str(self.bias))

                self.G.add_edge(j, "b " + str(self.bias))

        self.G.toggle_physics(True)

        self.G.show("network.html")
