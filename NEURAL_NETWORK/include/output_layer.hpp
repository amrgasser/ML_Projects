#ifndef __OUTPUT_LAYER_HPP
#define __OUTPUT_LAYER_HPP

#include "layer.hpp"
#include "data.hpp"

class OutputLayer : public Layer
{
public:
    OutputLayer(int prev, int curr) : Layer(prev, curr) {}
    void feed_forward(Layer);
    void back_prop(data *data);
    void update_weights(doube, Layer *);
};

#endif