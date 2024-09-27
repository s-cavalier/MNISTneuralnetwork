# My MNIST Neural Network
Very basic system designed to interpret the MNIST dataset through a neural network. <br>
Currently achieves ~94% accuracy, up to 99.99% on the validation (testing) set within about 30 epochs or so of the entire dataset. <br>
Network class is designed to be applicable to any neural network (i.e inputs and outputs vectors, all other handling such as using the brightest node is on user handling to maintain generalization/abstraction). <br>
Implements L2 regularization, dropout, log-loss cost function `(ylna + (1 - y)ln(1-a))`, and sigmoid neurons. <br>
Also uses the [Eigen API](https://gitlab.com/libeigen/eigen) for fast linear algebra methods.

For use just go to src and build with `make` and run the corresponding `init.exe` file.
As a library, remove the `init.exe` and `Input.h / Input.cpp` files. Those are intended exclusively to immediately load/read/use data.
Only Network.h and Network.cpp are needed for compilation and use.
