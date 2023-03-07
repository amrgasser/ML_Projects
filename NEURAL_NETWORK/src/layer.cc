#include "layer.hpp"

Layer::Layer(int prev, int curr)
{
    for (int i = 0; i < curr; i++)
    {
        this->neurons.push_back(new Neuron(prev));
    }
    this->current_layer_size = curr;
}