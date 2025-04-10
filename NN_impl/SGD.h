#pragma once
#include "Optimizer.h"
#include <Eigen/Core>

class SGD : public Optimizer {
public:
    SGD(double lr = 0.01) : Optimizer(lr) {}
    
    void update(
        Eigen::MatrixXd& weights,
        Eigen::VectorXd& biases,
        const Eigen::MatrixXd& weight_gradients,
        const Eigen::VectorXd& bias_gradients) override 
    {
        weights -= learning_rate * weight_gradients;
        biases -= learning_rate * bias_gradients;
    }
};