#ifndef N_NETWORK
#define N_NETWORK
#include <utility>
#include <vector>
#include "../Dependencies/eigen/Eigen/Dense"

class Network {
    //Sigmoid function
    float sig(const float &f) const;

    //Component-wise sigmoid function. Operates on the vector inplace.
    void sig_v(Eigen::VectorXf &v) const;

    //Componenet-wise sigmoid prime function. Returns copy of vector, does not modify original.
    Eigen::VectorXf sig_prime(Eigen::VectorXf v) const;

    // Stores the count of nodes in the input and output layers.
    // layers = hidden_layers + 1 = biases.size() = weights.size() = a.size() - 1 for indexing purposes (i.e. a[layers] is a last index of a).
    int input_nodes, output_nodes, layers;

    // List of matrices for learning.
    // Dimension = layers - 1
    std::vector<Eigen::MatrixXf> weights;

    // List of vectors for learning.
    // Dimension = layers - 1
    std::vector<Eigen::VectorXf> biases;


public:

    // Returns a constant reference to the weights.
    // For dimension purposes other than finding dim(x = a[0]), use biases.
    // If finding dim(x), dim(x) = rows(w[0])
    const std::vector<Eigen::MatrixXf>& get_weights() const;

    // Returns a constant reference to the biases.
    // Layer count = biases + 1
    // dim(a[i]) = dim(b[i - 1])
    const std::vector<Eigen::VectorXf>& get_biases() const;

    // Learning rate to scale gradient. Set by default to 0.15.
    float learning_rate;

    //For regularization, initially set to 1.
    float lambda = 1;

    //i defines the ith hidden layer and nodes_per_layer[i] defines the length of that hidden layer.
    Network(const int &input_layer_len, std::vector<unsigned int> nodes_per_layer, const int &output_layer_len);

    //Does a feedforward then a backpropagation.
    //Cost function is based on exp_output.
    Eigen::VectorXf train(const Eigen::VectorXf& data, const Eigen::VectorXf& exp_output);

    //Just does a feedforward for testing.
    Eigen::VectorXf activate(const Eigen::VectorXf& data) const;

};

#endif