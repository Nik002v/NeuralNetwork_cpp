#pragma once
#include "Layer.h"
#include <string>
#include <Eigen/Core>

class Dense : public Layer {
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    std::string activation;
    Eigen::VectorXd last_input;   
    Eigen::VectorXd last_output;  

public:
    Dense(int input_dim, int output_dim, std::string activation = "relu");

    Eigen::MatrixXd& get_weights() { return weights; }
    Eigen::VectorXd& get_biases() { return biases; }
    void set_weights(const Eigen::MatrixXd& new_weights) { weights = new_weights; }
    void set_biases(const Eigen::VectorXd& new_biases) { biases = new_biases; }

    Eigen::VectorXd forward(const Eigen::VectorXd& input) override;
    Gradients backward(const Eigen::VectorXd& gradient) override;

private:
    Eigen::VectorXd relu(const Eigen::VectorXd& x) const;
    Eigen::VectorXd relu_derivative(const Eigen::VectorXd& x) const;
    Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) const;    
    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& x) const;
    Eigen::VectorXd tanh(const Eigen::VectorXd& x) const;
    Eigen::VectorXd tanh_derivative(const Eigen::VectorXd& x) const;
    Eigen::VectorXd softmax(const Eigen::VectorXd& x) const;
    Eigen::VectorXd softmax_derivative(const Eigen::VectorXd& x) const;
    Eigen::VectorXd linear(const Eigen::VectorXd& x) const;
    Eigen::VectorXd linear_derivative(const Eigen::VectorXd& x) const;
    Eigen::VectorXd activation_function(const Eigen::VectorXd& x, std::string activation) const;
    Eigen::VectorXd activation_derivative(const Eigen::VectorXd& x, std::string activation) const;
};