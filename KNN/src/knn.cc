#include <cmath>
#include <limits>
#include <map>
#include <set>
#include "stdint.h"
#include "../../include/data_handler.hpp"
#include "../include/knn.hpp"

knn::knn(int val)
{
    k = val;
}
knn::knn() {}
knn::~knn() {}

void knn::find_knearest(data *query_point)
{
    neighbors = new std::vector<data *>;
    int index = 0;
    double min = std::numeric_limits<double>::max();
    double previous_min = min;
    for (int i = 0; i < k; i++)
    {
        if (i == 0)
        {
            for (int j = 0; j < training_data->size(); j++)
            {
                double distance = calculate_distance(query_point, training_data->at(j));
                training_data->at(j)->set_distance(distance);
                if (distance < min)
                {
                    min = distance;
                    index = j;
                }
            }
            neighbors->push_back(training_data->at(index));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        }
        else
        {
            for (int j = 0; j < training_data->size(); j++)
            {
                double distance = training_data->at(j)->get_distance();
                if (distance > previous_min && distance < min)
                {
                    min = distance;
                    index = j;
                }
            }
            neighbors->push_back(training_data->at(index));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        }
    }
}
void knn::set_training_data(std::vector<data *> *vect)
{
    training_data = vect;
}
void knn::set_test_data(std::vector<data *> *vect)
{
    test_data = vect;
}
void knn::set_validation_data(std::vector<data *> *vect)
{
    validation_data = vect;
}
void knn::set_k(int val)
{
    k = val;
}

int knn::predict()
{
    std::map<uint8_t, int> class_freq;

    for (int i = 0; i < neighbors->size(); i++)
    {
        if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end())
        {
            class_freq[neighbors->at(i)->get_label()] = 1;
        }
        else
        {
            class_freq[neighbors->at(i)->get_label()]++;
        }
    }

    int best = 0;
    int max = 0;

    for (auto kv : class_freq)
    {
        if (kv.second > max)
        {
            max = kv.second;
            best = kv.first;
        }
    }
    neighbors->clear();
    return best;
}
double knn::calculate_distance(data *query_point, data *input)
{
    double distance = 0.0;

#ifdef EUCLID
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
    {
        distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
        if (query_point->get_feature_vector_size() != input->get_feature_vector_size())
        {
            printf("Vector size mismatch.\n");
            exit(1);
        }
    }
    distance = sqrt(distance);
    return distance;
#elif defined MANHATTAN
    // MANHATTAN IMPL LATER
#endif
}
double knn::validate_performance()
{
    double current_performance = 0.0;
    int count = 0;
    int data_index = 0;
    for (data *query_point : *validation_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        if (prediction == query_point->get_label())
        {
            count++;
        }
        data_index++;
        current_performance = (double)count * 100.0 / (double)data_index;
        // printf("Current Performance = %.3f %%\n", current_performance);
    }
    printf("Validation Performance for K: %d = %.3f %%\n", k, current_performance);
    return current_performance;
}
double knn::test_performance()
{
    double current_performance = 0.0;
    int count = 0;
    int data_index = 0;
    for (data *query_point : *test_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        if (prediction == query_point->get_label())
        {
            count++;
        }
        data_index++;
        current_performance = (double)count * 100.0 / (double)data_index;
        // printf("Current Performance = %.3f %%\n", current_performance);
    }
    printf("Test Performance for K: %d  = %.3f %%\n", k, current_performance);
    return current_performance;
}

int main()
{
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../train-images-idx3-ubyte");
    dh->read_feature_labels("../train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    knn *knearest = new knn(1);
    knearest->set_training_data(dh->get_training_data());
    knearest->set_test_data(dh->get_test_data());
    knearest->set_validation_data(dh->get_validation_data());
    double performance = 0.0;
    double best_performance = 0.0;
    int best_k = 1;
    for (int i = 1; i <= 10; i++)
    {
        knearest->set_k(i);
        performance = knearest->validate_performance();
        if (performance > best_performance)
        {
            best_performance = performance;
            best_k = i;
        }
        printf("KNearest performance with best K: %d = %.3f %%.\n", best_k, knearest->test_performance());
    }
    knearest->set_k(best_k);
    printf("KNearest performance with best K: %d = %.3f %%.\n", best_k, knearest->test_performance());
}