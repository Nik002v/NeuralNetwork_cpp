#include "Adam.h"
#include <cmath>

Adam::Adam(double lr, double b1, double b2, double eps)
    : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Adam::update(
    Eigen::MatrixXd& weights,
    Eigen::VectorXd& biases,
    const Eigen::MatrixXd& weight_gradients,
    const Eigen::VectorXd& bias_gradients) 
{
    t++;

    if (m_weights.rows() == 0) {
        m_weights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
        v_weights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
        m_biases = Eigen::VectorXd::Zero(biases.size());
        v_biases = Eigen::VectorXd::Zero(biases.size());
    }

    m_weights = beta1 * m_weights + (1.0 - beta1) * weight_gradients;
    v_weights = beta2 * v_weights + (1.0 - beta2) * weight_gradients.array().square().matrix();
    
    m_biases = beta1 * m_biases + (1.0 - beta1) * bias_gradients;
    v_biases = beta2 * v_biases + (1.0 - beta2) * bias_gradients.array().square().matrix();

    double m_scale = 1.0 / (1.0 - std::pow(beta1, t));
    double v_scale = 1.0 / (1.0 - std::pow(beta2, t));

    weights -= learning_rate * 
        (m_weights.array() * m_scale / 
        ((v_weights.array() * v_scale).sqrt() + epsilon)).matrix();
    
    biases -= learning_rate * 
        (m_biases.array() * m_scale / 
        ((v_biases.array() * v_scale).sqrt() + epsilon)).matrix();
}
