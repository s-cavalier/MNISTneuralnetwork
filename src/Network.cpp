#include "Network.h"
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>
#include <list>
#include <iterator>
#include <random>

const std::vector<Eigen::MatrixXf>& Network::get_weights() const {
    return weights;
}

const std::vector<Eigen::VectorXf>& Network::get_biases() const {
    return biases;
}

float Network::sig(const float &f) const {
    return 1.0 / (1 + std::exp(-f));
}

//Sigmoid function
void Network::sig_v(Eigen::VectorXf &v) const {
    for (int i = 0; i < v.rows(); i++) v(i) = sig(v(i));
}

//d/dx Sigmoid Function
Eigen::VectorXf Network::sig_prime(Eigen::VectorXf v) const {
    for (int i = 0; i < v.rows(); i++) v(i) = sig(v(i))*(1-sig(v(i)));
    return v;
}


Network::Network(const int &input_layer_len, std::vector<unsigned int> nodes_per_layer, const int &output_layer_len) {
    learning_rate = 0.15;
    input_nodes = input_layer_len;
    layers = 1 + nodes_per_layer.size();
    //width needs to be 784
    //height = next layer
    nodes_per_layer.push_back(output_layer_len);
    int prev_len = input_layer_len;

    for (int i = 0; i < nodes_per_layer.size(); i++) {
        std::srand(time(0));
        weights.push_back(Eigen::MatrixXf::Random(nodes_per_layer[i], prev_len) * 2);
        biases.push_back(Eigen::VectorXf::Random(nodes_per_layer[i]) * 2);

        prev_len = nodes_per_layer[i];
    }

}

Eigen::VectorXf Network::activate(const Eigen::VectorXf& data) const {
    Eigen::VectorXf active = data;
    for (int i = 0; i < layers; i++) {
        active = weights[i] * active + biases[i];
        sig_v(active);
    }

    return active;
}

Eigen::VectorXf Network::train(const Eigen::VectorXf& data, const Eigen::VectorXf& exp_output) {
    //Store activation layers
    //Pre-sigmoid
    std::vector<Eigen::VectorXf> z;
    //Post-sigmoid
    std::vector<Eigen::VectorXf> a;

    //Randomly set nodes to zero
    std::random_device rd;
    std::uniform_int_distribution dist(0, 1);

    // feedforward and save activation layers
    a.push_back(data);
    for (int i = 0; i < layers; i++) {
        Eigen::VectorXf active = a[i];
        active = weights[i] * active + biases[i];
        if (i < (layers - 1)) { for (float f : active) f *= dist(rd); }
        z.push_back(active);
        sig_v(active);
        a.push_back(active);
    }

    //Compute final layer L error vector
    std::list<Eigen::VectorXf> errors(layers);

    // Uses list to push to front, iterator is necessary for access
    auto rit = errors.rbegin();
    *rit = a[layers] - exp_output;
    rit++;
    int i = layers - 2;

    // Core backpropagation
    // Come back and double check math related to indexing, seems error-prone ??????
    for (rit; rit != errors.rend(); rit++) {
        *rit = (weights[i + 1].transpose() * ( *std::prev(rit) )).array() * sig_prime(z[i]).array();
        i--;
    }

    // Non-stochastic gradient descent
    auto it = errors.begin();
    i = 0;
    while (i < layers) {
        // Biases gradient
        biases[i] -= (*it) * learning_rate;

        // Weights gradient
        for (int j = 0; j < weights[i].rows(); j++) {
            for (int k = 0; k < weights[i].cols(); k++) {
                weights[i](j, k) = (1 - learning_rate * lambda) * weights[i](j, k) - learning_rate * a[i](k) * (*it)(j);
            }
        }

        it++;
        i++;
    }

    return a[layers];

}