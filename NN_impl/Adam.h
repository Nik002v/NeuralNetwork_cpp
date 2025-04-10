#pragma once
#include "Optimizer.h"

class Adam : public Optimizer {
private:
    double beta1;
    double beta2;
    double epsilon;
    Eigen::MatrixXd m_weights;
    Eigen::VectorXd m_biases;
    Eigen::MatrixXd v_weights;
    Eigen::VectorXd v_biases;
    int t;

public:
    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);
    
    void update(
        Eigen::MatrixXd& weights,
        Eigen::VectorXd& biases,
        const Eigen::MatrixXd& weight_gradients,
        const Eigen::VectorXd& bias_gradients) override;
};