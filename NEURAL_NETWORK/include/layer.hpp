#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "neuron.hpp"
#include <set>

class Layer
{
private:
    /* data */
public:
    Layer(int, int);
    ~Layer();

    int current_layer_size;
    std::vector<Neuron *> neurons;
    std::vector<double> layer_outputs;
};

#endif // !__LAYER_HPP