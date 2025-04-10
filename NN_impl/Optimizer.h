#pragma once
#include <Eigen/Core>

class Optimizer {
protected:
    double learning_rate;

public:
    Optimizer(double lr = 0.001);
    virtual ~Optimizer() = default;
    
    virtual void update(
        Eigen::MatrixXd& weights,
        Eigen::VectorXd& biases,
        const Eigen::MatrixXd& weight_gradients,
        const Eigen::VectorXd& bias_gradients) = 0;
};