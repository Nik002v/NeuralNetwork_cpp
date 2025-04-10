#pragma once
#include <memory>
#include <Eigen/Core>
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"


class NeuralNetwork {
public:
    virtual ~NeuralNetwork() = default;
    virtual void train(const std::vector<Eigen::VectorXd>& X,
                      const std::vector<Eigen::VectorXd>& y,
                      int epochs,
                      int batch_size) = 0;
    virtual Eigen::VectorXd predict(const Eigen::VectorXd& input) = 0;
    virtual double eval(const std::vector<Eigen::VectorXd>& X,
                       const std::vector<Eigen::VectorXd>& y) = 0;
};


class Sequential : public NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimizer> optimizer;
    std::unique_ptr<Loss> loss_function;
    int size;
    int num_weights;
    
    Eigen::VectorXd feedforward(const Eigen::VectorXd& input);

public:
    Sequential();

    void add(std::unique_ptr<Layer> layer);
    void set_loss(std::unique_ptr<Loss> loss) {
        loss_function = std::move(loss);
    }
    Eigen::VectorXd predict(const Eigen::VectorXd& input) override;
    double eval(const std::vector<Eigen::VectorXd>& X,
               const std::vector<Eigen::VectorXd>& y) override;
    void train(const std::vector<Eigen::VectorXd>& X,
               const std::vector<Eigen::VectorXd>& y,
               int epochs = 10,
               int batch_size = 32) override;
};
