#include "network.hpp"
#include "layer.hpp"
#include "data_handler.hpp"
#include <numeric>

Network::Network(std::vector<int> spec, int input_size, int num_classes, double learning_rate)
{
    for (int i = 0; i < spec.size(); i++)
    {
        if (i == 0)
            layers.push_back(new Layer(input_size, spec.at(i)));
        else
            layers.push_back(new Layer(layers.at(i - 1)->neurons.size(), spec.at(i)));
    }
    layers.push_back(new Layer(layers.at(layers.size() - 1)->neurons.size(), num_classes));
    this->learning_rate = learning_rate;
}
Network::~Network() {}

double Network::activate(std::vector<double> weights, std::vector<double> inputs)
{
    double activation = weights.back(); // bias

    for (int i = 0; i < weights.size() - 1; i++)
    {
        activation += weights[i] * inputs[i];
    }

    return activation;
}

double Network::transfer(double activation)
{
    return 1.0 / (1.0 + exp(-activation));
}

double Network::transfer_derivative(double output)
{
    return output * (1 - output);
}

std::vector<double> Network::fprop(data *data)
{
    std::vector<double> inputs = *data->get_double_feature_vector();

    for (int i = 0; i < layers.size(); i++)
    {
        Layer *layer = layers.at(i);
        std::vector<double> new_inputs;
        for (Neuron *n : layer->neurons)
        {
            double activation = this->activate(n->weights, inputs);
            n->output = this->transfer(activation);
            new_inputs.push_back(n->output);
        }
        inputs = new_inputs;
    }

    // returning the output of the output layer (prediction)
    return inputs;
}

// chain rule
//  backprop from output to input while calculating error
//  contribution of the previous layer to the current layer's output error
void Network::bprop(data *data)
{
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        Layer *layer = layers.at(i);
        std::vector<double> errs;
        if (i != layers.size() - 1)
        {
            for (int j = 0; j < layer->neurons.size(); j++)
            {
                double error = 0.0;
                for (Neuron *n : layers.at(i + 1)->neurons)
                {
                    // accumulating this current neurons contrib to the error to all neurons in next layer
                    error += (n->weights.at(j) * n->delta);
                }
                errs.push_back(error);
            }
        }
        else
        {
            for (int j = 0; j < layer->neurons.size(); j++)
            {
                Neuron *n = layer->neurons.at(j);
                errs.push_back((double)data->get_class_vector()->at(j) - n->output);
            }
        }
        for (int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron *n = layer->neurons.at(j);
            n->delta = errs.at(j) * this->transfer_derivative(n->output); // gradient descent part of back_prop
        }
    }
}

void Network::update_weights(data *data)
{
    std::vector<double> inputs = *data->get_double_feature_vector();

    for (int i = 0; i < layers.size(); i++)
    {
        if (i != 0)
        {
            for (Neuron *n : layers.at(i - 1)->neurons)
            {
                inputs.push_back(n->output);
            }
        }
        for (Neuron *n : layers.at(i)->neurons)
        {
            for (int j = 0; j < inputs.size(); j++)
            {
                n->weights.at(j) += this->learning_rate * n->delta * inputs.at(j);
            }
            n->weights.back() += this->learning_rate * n->delta;
        }

        inputs.clear();
    }
}

int Network::predict(data *data)
{
    std::vector<double> outputs = fprop(data);

    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int num_epochs)
{
    for (int i = 0; i < num_epochs; i++)
    {
        double sum_error = 0.0;
        for (data *d : *this->training_data)
        {
            std::vector<double> outputs = fprop(d);
            std::vector<int> expected = *d->get_class_vector();
            double temp_sum = 0.0;
            for (int j = 0; j < outputs.size(); j++)
            {
                temp_sum += pow((double)expected.at(j) - outputs.at(j), 2);
            }
            sum_error += temp_sum;
            bprop(d);
            update_weights(d);
        }
        printf("Iteration: %d \t Error = %.4f\n", i, sum_error);
    }
}
double Network::test()
{
    double num_correct = 0.0;
    double count = 0.0;

    for (data *data : *this->test_data)
    {
        count++;
        int index = predict(data);
        if (data->get_class_vector()->at(index) == 1)
            num_correct++;
    }

    return num_correct / count;
}
void Network::validate()
{
    double num_correct = 0.0;
    double count = 0.0;

    for (data *data : *this->validation_data)
    {
        count++;
        int index = predict(data);
        if (data->get_class_vector()->at(index) == 1)
            num_correct++;
    }
    printf("Validation Performance: %.4f\n", num_correct / count);
}

int main()
{

    data_handler *dh = new data_handler();
    // #ifdef MNIST
    //     dh->read_input_data("../train-images-idx3-ubyte");
    //     dh->read_label_data("../train-labels-idx1-ubyte");
    //     dh->count_classes();
    // #else
    // #endif // DEBUG

    dh->read_csv("../iris.csv", ",");
    dh->split_data();
    std::vector<int> hidden_layers = {10};
    auto lambda = [&]()
    {
        Network *net = new Network(
            hidden_layers,
            dh->get_training_data()->at(0)->get_double_feature_vector()->size(),
            dh->get_class_counts(),
            0.25);
        net->set_training_data(dh->get_training_data());
        net->set_test_data(dh->get_test_data());
        net->set_validation_data(dh->get_validation_data());
        net->train(15);
        net->validate();
        printf("Test Performance: %.4f\n", net->test());
    };
    lambda();

    return 0;
}