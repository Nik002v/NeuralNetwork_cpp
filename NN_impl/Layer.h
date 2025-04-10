#pragma once
#include <Eigen/Core>

class Layer {
protected:
    int input_dim;
    int output_dim;

public:
    struct Gradients {
        Eigen::MatrixXd weight_gradients;
        Eigen::VectorXd bias_gradients;
        Eigen::VectorXd input_gradients;
    };

    Layer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {}
    virtual ~Layer() = default;

    virtual Eigen::MatrixXd& get_weights()  = 0;
    virtual Eigen::VectorXd& get_biases()  = 0;
    
    virtual Eigen::VectorXd forward(const Eigen::VectorXd& input) = 0;
    virtual Gradients backward(const Eigen::VectorXd& gradient) = 0;
};