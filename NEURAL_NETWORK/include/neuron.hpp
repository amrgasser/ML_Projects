#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <cmath>
#include <set>

class Neuron
{
public:
    Neuron(int);
    ~Neuron();

    double output;
    double delta;
    std::vector<double> weights;
    void initialize_weights(int);
    // Setters
};

#endif