# My MNIST Neural Network
Very basic system designed to interpret the MNIST dataset through a neural network. <br>
Currently achieves ~94% accuracy.
Network class is designed to be applicable to any neural network (i.e inputs and outputs vectors, all other handling such as using the brightest node is on user handling to maintain generalization/abstraction).
Implements L2 regularization, dropout, log-loss cost function `(ylna + (1 - y)ln(1-a))`, and sigmoid neurons.
Also uses the [Eigen API](https://gitlab.com/libeigen/eigen) for fast linear algebra methods.
