#include "Dense.h"
#include <random>

Dense::Dense(int input_dim, int output_dim, std::string activation)
    : Layer(input_dim, output_dim), activation(activation) {
    
    weights = Eigen::MatrixXd::Random(output_dim, input_dim) * std::sqrt(2.0 / input_dim);
    biases = Eigen::VectorXd::Zero(output_dim);
}

Eigen::VectorXd Dense::forward(const Eigen::VectorXd& input) {
    last_input = input;
    last_output = weights * input + biases;
    return activation_function(last_output, activation);
}

Layer::Gradients Dense::backward(const Eigen::VectorXd& gradient) {
    Eigen::VectorXd d_activation = activation_derivative(last_output, activation);
    Eigen::VectorXd d_preactivation = gradient.array() * d_activation.array();
    
    Layer::Gradients grads;
    grads.weight_gradients = d_preactivation * last_input.transpose();
    grads.bias_gradients = d_preactivation;
    grads.input_gradients = weights.transpose() * d_preactivation;
    
    return grads;
}

Eigen::VectorXd Dense::activation_function(const Eigen::VectorXd& x, std::string activation) const {
    if (activation == "relu") return relu(x);
    if (activation == "sigmoid") return sigmoid(x);
    if (activation == "tanh") return tanh(x);
    if (activation == "softmax") return softmax(x);
    return linear(x); 
}

Eigen::VectorXd Dense::activation_derivative(const Eigen::VectorXd& x, std::string activation) const {
    if (activation == "relu") return relu_derivative(x);
    if (activation == "sigmoid") return sigmoid_derivative(x);
    if (activation == "tanh") return tanh_derivative(x);
    if (activation == "softmax") return softmax_derivative(x);
    return linear_derivative(x); 
}

Eigen::VectorXd Dense::relu(const Eigen::VectorXd& x) const {
    return x.array().max(0.0);
}

Eigen::VectorXd Dense::relu_derivative(const Eigen::VectorXd& x) const {
    return (x.array() > 0.0).cast<double>();
}

Eigen::VectorXd Dense::sigmoid(const Eigen::VectorXd& x) const {
    return 1.0 / (1.0 + (-x.array()).exp());
}

Eigen::VectorXd Dense::sigmoid_derivative(const Eigen::VectorXd& x) const {
    Eigen::ArrayXd sig = sigmoid(x);
    return sig * (1.0 - sig);
}

Eigen::VectorXd Dense::tanh(const Eigen::VectorXd& x) const {
    return x.array().tanh();
}

Eigen::VectorXd Dense::tanh_derivative(const Eigen::VectorXd& x) const {
    auto th = tanh(x);
    return 1.0 - th.array().square();
}

Eigen::VectorXd Dense::softmax(const Eigen::VectorXd& x) const {
    Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp(); 
    return exp_x.array() / exp_x.sum();
}

Eigen::VectorXd Dense::softmax_derivative(const Eigen::VectorXd& x) const {
    Eigen::ArrayXd s = softmax(x);
    return s * (1.0 - s);
}

Eigen::VectorXd Dense::linear(const Eigen::VectorXd& x) const {
    return x;
}

Eigen::VectorXd Dense::linear_derivative(const Eigen::VectorXd& x) const {
    return Eigen::VectorXd::Ones(x.size());
}

