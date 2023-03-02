#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "hidden_layer.hpp"
#include "input_layer.hpp"
#include "output_layer.hpp"
#include "common.hpp"

class Network : public common_data
{
public:
    std::vector<Layer *> layers;
    double learning_rate;
    double test_performance;

    Network(std::vector<int> spec, int, int, double).~Netowkr();

    std::vector<double> fprop(data data);
    double activate(std::vector<double>, std::vector<double>);
    double transfer(double);
    double transfer_derivative(double); // back prop

    // back prop methods
    void bprop(data *data);
    void update_weights(data *data);
    int predict(data *data); // return ind of max value;

    void traing(int); // num of iterations
    double test();
    void validate();
};

#endif