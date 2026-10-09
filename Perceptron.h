//
// Created by Tim Healey on 10/8/26.
//

#ifndef RANDOM_ML_PERCEPTRON_H
#define RANDOM_ML_PERCEPTRON_H
#include <vector>
#include <Eigen/Dense>

class Perceptron {
public:
    Perceptron() : eta(0.01), n_iter(50), rand_state(1) {};
    void fit(Eigen::ArrayXXd& X, Eigen::ArrayXd& y);
private:
    double predict(Eigen::ArrayXd& x);
    double net_input(Eigen::ArrayXd& x);
    Eigen::ArrayXd weights;
    std::vector<int> errors;
    // learning rate
    double eta;
    double bias = 0.0;
    // iterations
    int n_iter;
    int rand_state;
};


#endif //RANDOM_ML_PERCEPTRON_H