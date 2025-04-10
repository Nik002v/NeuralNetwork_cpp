#pragma once
#include <Eigen/Core>

class Loss {
public:
    virtual ~Loss() = default;
    virtual double compute(const Eigen::VectorXd& y_pred, 
                         const Eigen::VectorXd& y_true) const = 0;
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd& y_pred, 
                                   const Eigen::VectorXd& y_true) const = 0;
};

class MSE : public Loss {
public:
    double compute(const Eigen::VectorXd& y_pred, 
                  const Eigen::VectorXd& y_true) const override {
        return (y_pred - y_true).squaredNorm() / 2.0;
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& y_pred, 
                           const Eigen::VectorXd& y_true) const override {
        return y_pred - y_true;
    }
};

class CrossEntropy : public Loss {
public:
    double compute(const Eigen::VectorXd& y_pred, 
                  const Eigen::VectorXd& y_true) const override {
        return -(y_true.array() * y_pred.array().log()).sum();
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& y_pred, 
                           const Eigen::VectorXd& y_true) const override {
        return -y_true.array() / y_pred.array();
    }
};