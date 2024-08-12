// STD Imports
#include <iostream>
#include <iomanip>

// Local Dependencies
#include "Input.h"
#include "Network.h"

// Imports
#include "../Dependencies/eigen/Eigen/Dense"

Eigen::VectorXf stdvec28x28_to_eigenvec784(const std::vector<std::vector<unsigned char>> &vec) {
    Eigen::VectorXf v(784);
    int v_index = 0;
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            v(v_index++) = ( ((float)vec[r][c]) / 255);
        }
    }
    return v;
}

int main(int argc, char** argv) {

    //Load training set
    std::cout << "Generating training data." << std::endl;
    MNIST_loader training_set("Data/MNIST/train-images-idx3-ubyte", "Data/MNIST/train-labels-idx1-ubyte", 60000);
    std::cout << "Generating testing data." << std::endl;
    MNIST_loader testing_set("Data/MNIST/t10k-images-idx3-ubyte", "Data/MNIST/t10k-labels-idx1-ubyte", 10000);
    
    //Activate network
    std::cout << "Done. Starting network..." << std::endl;
    Network network(784, { 225 }, 10);
    network.learning_rate = 0.15;
    network.lambda = 5.0 / training_set.size();
    

    // Train system for each image in training set
    for (int i = 0; i < training_set.size(); i++) {
        std::cout << "Training: Processing image " << i + 1 << " / " << training_set.size() << std::endl; 
        
        // Set expected vector for cost function
        Eigen::VectorXf expected(10);
        for (int j = 0; j < 10; j++) expected(j) = (j == (int)training_set[i].second);

        // Run and show results.
        Eigen::VectorXf output = network.train(stdvec28x28_to_eigenvec784(training_set[i].first), expected);

        float max = 0;
        int max_ind = 0;
        for (int j = 0; j < 10; j++) {
            if (max >= output(j)) continue;
            max = output(j);
            max_ind = j;
        }

        std::cout << "Expected: " << (int)training_set[i].second << ", Recieved: " << max_ind << ", Success? " << std::boolalpha << (max_ind == (int)training_set[i].second) << std::endl << std::endl;
    }

    int correct = 0;

    // Test system for each image in testing set
    for (int i = 0; i < testing_set.size(); i++) {
         std::cout << "Testing: Processing image " << i + 1 << " / " << testing_set.size() << std::endl; 

        // Run and show results.
        Eigen::VectorXf output = network.activate(stdvec28x28_to_eigenvec784(training_set[i].first));

        float max = 0;
        int max_ind = 0;
        for (int j = 0; j < 10; j++) {
            if (max >= output(j)) continue;
            max = output(j);
            max_ind = j;
        }

        bool check = (max_ind == (int)training_set[i].second);
        correct += check;

        std::cout << "Expected: " << (int)training_set[i].second << ", Recieved: " << max_ind << ", Success? " << std::boolalpha << check << ", Accuracy: " << 100.0 * ((float)correct) / (i + 1) << "%" << std::endl << std::endl;

    }
    
    std::cout << "Finished with no errors, accuracy rate: " << 100.0 * ((float)correct) / testing_set.size() << "%" << std::endl;

    return 0;
}