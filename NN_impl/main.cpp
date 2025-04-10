#include "NeuralNetwork.h"
#include "Dense.h"
#include "Loss.h"
#include "SGD.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    // Random number generation for dataset creation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 0.3);

    // Generate synthetic classification data
    const int samples_per_class = 100;
    const int num_classes = 3;
    const int input_dim = 2;
    
    // Create data matrices
    Eigen::MatrixXd X(samples_per_class * num_classes, input_dim);
    Eigen::MatrixXd Y(samples_per_class * num_classes, num_classes);

    // Generate three clusters of points
    double angle = 2.0 * 3.1415 / num_classes;
    for (int c = 0; c < num_classes; c++) {
        double center_x = 2.0 * std::cos(c * angle);
        double center_y = 2.0 * std::sin(c * angle);
        
        for (int i = 0; i < samples_per_class; i++) {
            int idx = c * samples_per_class + i;
            // Add noise to circular pattern
            X(idx, 0) = center_x + dist(gen);
            X(idx, 1) = center_y + dist(gen);
            
            // One-hot encoding
            Y.row(idx) = Eigen::VectorXd::Zero(num_classes);
            Y(idx, c) = 1.0;
        }
    }

    // Convert to vector format for network
    std::vector<Eigen::VectorXd> X_train;
    std::vector<Eigen::VectorXd> y_train;
    for (int i = 0; i < X.rows(); i++) {
        X_train.push_back(X.row(i));
        y_train.push_back(Y.row(i));
    }

    // Create model
    Sequential model;
    
    // Add layers
    model.add(std::make_unique<Dense>(input_dim, 32, "relu"));
    model.add(std::make_unique<Dense>(32, 16, "relu"));
    model.add(std::make_unique<Dense>(16, num_classes, "softmax"));
    
    // Set loss function
    model.set_loss(std::make_unique<CrossEntropy>());
    
    // Train model
    std::cout << "Training model on " << X_train.size() << " samples...\n";
    model.train(X_train, y_train, 100, 32);  // epochs=100, batch_size=32
    
    // Test on some sample points
    std::cout << "\nTesting model:\n";
    std::vector<Eigen::Vector2d> test_points = {
        {2.0, 0.0},   // Should be class 0
        {-1.0, 1.7},  // Should be class 1
        {-1.0, -1.7}  // Should be class 2
    };

    for (const auto& point : test_points) {
        Eigen::VectorXd pred = model.predict(point);
        std::cout << "Input: [" << point.transpose() << "]\n";
        std::cout << "Probabilities: [" << pred.transpose() << "]\n";
        std::cout << "Predicted class: " << pred.maxCoeff() << "\n\n";
    }
    

    // Prepare data for visualization
    std::vector<double> x0, y0, x1, y1, x2, y2;

    for (int i = 0; i < X.rows(); i++) {
        if (Y(i, 0) == 1.0) {
            x0.push_back(X(i, 0));
            y0.push_back(X(i, 1));
        } else if (Y(i, 1) == 1.0) {
            x1.push_back(X(i, 0));
            y1.push_back(X(i, 1));
        } else {
            x2.push_back(X(i, 0));
            y2.push_back(X(i, 1));
        }
    }

    // Plot training data
    plt::figure_size(10, 10);
    plt::scatter(x0, y0, 50, {{"c", "red"}, {"label", "Class 0"}});
    plt::scatter(x1, y1, 50, {{"c", "blue"}, {"label", "Class 1"}});
    plt::scatter(x2, y2, 50, {{"c", "green"}, {"label", "Class 2"}});
    
    // Plot decision boundary
    const int grid_points = 100;
    std::vector<double> xx, yy;
    std::vector<int> classes;
    
    for (double x = -3; x <= 3; x += 0.06) {
        for (double y = -3; y <= 3; y += 0.06) {
            Eigen::Vector2d point(x, y);
            Eigen::VectorXd pred = model.predict(point);
            int predicted_class;
            pred.maxCoeff(&predicted_class);
            
            xx.push_back(x);
            yy.push_back(y);
            classes.push_back(predicted_class);
        }
    }
    
    // Create color map for decision boundary
    std::vector<double> colors(classes.size());
    for (size_t i = 0; i < classes.size(); i++) {
        colors[i] = static_cast<double>(classes[i]) / 2.0;
    }
    
    plt::scatter(xx, yy, 1, colors, {{"alpha", "0.2"}, {"cmap", "rainbow"}});
    plt::legend();
    plt::title("Classification Results with Decision Boundaries");
    plt::save("classification_plot.png");
    plt::show();

    return 0;
}