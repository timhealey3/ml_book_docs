#include "Perceptron.h"

#include <random>

void Perceptron::fit(Eigen::ArrayXXd& X, Eigen::ArrayXd& y) {
    const int n_samples  = X.rows();
    const int n_features = X.cols();

    // Initialize weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.01);

    weights.resize(n_features);               // important!
    for (int i = 0; i < n_features; ++i) {
        weights(i) = dist(gen);
    }

    bias = 0.0;
    errors.clear();

    for (int epoch = 0; epoch < n_iter; ++epoch) {
        int error = 0;

        for (int i = 0; i < n_samples; ++i) {
            Eigen::ArrayXd xi = X.row(i);     // now both double → OK
            double target = y(i);

            double update = eta * (target - predict(xi));

            weights += update * xi;
            bias    += update;

            error += (update != 0.0);
        }
        errors.push_back(error);
    }
}

double Perceptron::net_input(Eigen::ArrayXd& x) {
    return (x * weights).sum() + bias;
}

double Perceptron::predict(Eigen::ArrayXd& x) {
    return net_input(x) >= 0.0 ? 1.0 : 0.0;
}