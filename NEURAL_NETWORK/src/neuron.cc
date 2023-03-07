#include "neuron.hpp"
#include <random>

double generate_random_number(double min, double max)
{
    double random = (double)rand() / max;
    return min + random * (max - min);
}

Neuron::Neuron(int prev_layer_size)
{
    initialize_weights(prev_layer_size);
}

Neuron::~Neuron() {}

void Neuron::initialize_weights(int prev_layer_size)
{
    std::default_random_engine generateor();
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i <= prev_layer_size; i++)
    {
        weights.push_back(generate_random_number(-1.0, 1.0));
    }
}