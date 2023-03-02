#ifndef __HIDDEN_LAYER_HPP
#define __HIDDEN_LAYER_HPP

#include "layer.hpp"
#include "data.hpp"

class HiddenLayer : public Layer
{
public:
    HiddenLayer(int prev, int curr) : Layer(prev, curr) {}
    void feed_forward(Layer prev);
    void back_prop(Layer next);
    void update_weights(doube, Layer *);
};
#endif