#include "NeuralNetwork.h"
#include "SGD.h"
#include "Dense.h"

#include <random>
#include <numeric>
#include <iostream>


Sequential::Sequential() : size(0), num_weights(0) {
    optimizer = std::make_unique<SGD>();  
    loss_function = std::make_unique<MSE>();
}

void Sequential::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
    size++;
}

Eigen::VectorXd Sequential::feedforward(const Eigen::VectorXd& input) {
    Eigen::VectorXd current = input;
    for (const auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

Eigen::VectorXd Sequential::predict(const Eigen::VectorXd& input) {
    return feedforward(input);
}

double Sequential::eval(const std::vector<Eigen::VectorXd>& X,
                       const std::vector<Eigen::VectorXd>& y) {
    double loss = 0.0;
    for (size_t i = 0; i < X.size(); i++) {
        Eigen::VectorXd pred = predict(X[i]);
        loss += (pred - y[i]).squaredNorm();
    }
    return loss / X.size();
}


void Sequential::train(const std::vector<Eigen::VectorXd>& X,
                      const std::vector<Eigen::VectorXd>& y,
                      int epochs,
                      int batch_size) 
{
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        
        std::vector<size_t> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
        

        for (size_t i = 0; i < X.size(); i += batch_size) {
            size_t batch_end = std::min(i + batch_size, X.size());
            
            Layer::Gradients batch_grads;
            batch_grads.weight_gradients = Eigen::MatrixXd::Zero(layers.back()->get_weights().rows(), 
                                                               layers.back()->get_weights().cols());
            batch_grads.bias_gradients = Eigen::VectorXd::Zero(layers.back()->get_biases().size());
            

            for (size_t b = i; b < batch_end; b++) {
                size_t idx = indices[b];  
                
                Eigen::VectorXd output = feedforward(X[idx]);
            
                epoch_loss += loss_function->compute(output, y[idx]);
                Eigen::VectorXd gradient = loss_function->gradient(output, y[idx]);
                
                for (int j = layers.size() - 1; j >= 0; j--) {
                    Layer::Gradients grads = layers[j]->backward(gradient);
                    
                    batch_grads.weight_gradients += grads.weight_gradients;
                    batch_grads.bias_gradients += grads.bias_gradients;
                    
                    gradient = grads.input_gradients;
                }
            }
            
            size_t batch_size_actual = batch_end - i;
            batch_grads.weight_gradients /= batch_size_actual;
            batch_grads.bias_gradients /= batch_size_actual;
            
            auto* dense = dynamic_cast<Dense*>(layers.back().get());
            if (dense) {
                optimizer->update(dense->get_weights(),
                                dense->get_biases(),
                                batch_grads.weight_gradients,
                                batch_grads.bias_gradients);
            }
        }
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch 
                     << ", Loss: " << epoch_loss / X.size() 
                     << std::endl;
        }
    }
}