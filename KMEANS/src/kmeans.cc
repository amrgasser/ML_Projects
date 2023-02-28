#include "../include/kmeans.hpp"
#include <set>

kmeans::kmeans(int k)
{
    num_clusters = k;
    clusters = new std::vector<cluster_t *>;
    used_indexes = new std::unordered_set<int>;
}
void kmeans::init_clusters()
{
    for (int i = 1; i < num_clusters; i++)
    {
        int index = (rand() % training_data->size());
        while (used_indexes->find(index) != used_indexes->end())
        {
            index = (rand() % training_data->size());
        }
        clusters->push_back(new cluster(training_data->at(index)));
        used_indexes->insert(index);
    }
}
void kmeans::init_clusters_for_each_class()
{
    std::unordered_set<int> classes_used;
    for (int i = 0; i < training_data->size(); i++)
    {
        if (classes_used.find(training_data->at(i)->get_label()) == classes_used.end())
        {
            clusters->push_back(new cluster(training_data->at(i)));
            classes_used.insert(training_data->at(i)->get_label());
            used_indexes->insert(i);
        }
    }
}
void kmeans::train()
{
    while (used_indexes->size() < training_data->size())
    {
        int index = (rand() % training_data->size());
        while (used_indexes->find(index) != used_indexes->end())
        {
            index = (rand() % training_data->size());
        }
        double min_distance = std::numeric_limits<double>::max();
        int best_cluster = 0;

        for (int i = 0; i < clusters->size(); i++)
        {
            double current_dist = euclidean_distance(clusters->at(i)->centroid, training_data->at(index));
            if (current_dist < min_distance)
            {
                min_distance = current_dist;
                best_cluster = i;
            }
        }
        clusters->at(best_cluster)->add_to_cluster(training_data->at(index));
        used_indexes->insert(index);
    }
}
double kmeans::euclidean_distance(std::vector<double> *centroid, data *point)
{
    double distance = 0;
    for (int i = 0; i < centroid->size(); i++)
    {
        distance += pow(centroid->at(i) - point->get_feature_vector()->at(i), 2);
    }
    return sqrt(distance);
}
double kmeans::validate()
{
    double num_correct = 0.0;
    for (auto query_point : *validation_data)
    {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < clusters->size(); j++)
        {
            double distance = euclidean_distance(clusters->at(j)->centroid, query_point);
            if (distance < min_dist)
            {
                min_dist = distance;
                best_cluster = j;
            }
        }
        if (clusters->at(best_cluster)->most_frequent_class == query_point->get_label())
            num_correct++;
    }

    return 100 * (num_correct / (double)validation_data->size());
}
double kmeans::test()
{
    double num_correct = 0.0;
    for (auto query_point : *test_data)
    {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < clusters->size(); j++)
        {
            double distance = euclidean_distance(clusters->at(j)->centroid, query_point);
            if (distance < min_dist)
            {
                min_dist = distance;
                best_cluster = j;
            }
        }
        if (clusters->at(best_cluster)->most_frequent_class == query_point->get_label())
            num_correct++;
    }

    return 100 * (num_correct / (double)test_data->size());
}

int main()
{
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../train-images-idx3-ubyte");
    dh->read_feature_labels("../train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    double performance = 0.0;
    double best_performance = 0.0;
    int best_k;
    printf("%d.\n", dh->get_class_counts());
    for (int k = dh->get_class_counts(); dh->get_training_data()->size() * 0.1; k++)
    {
        kmeans *km = new kmeans(k);
        km->set_training_data(dh->get_training_data());
        km->set_test_data(dh->get_test_data());
        km->set_validation_data(dh->get_validation_data());

        km->init_clusters();
        printf("Done Init @k : %d.\n", k);
        km->train();
        performance = km->validate();

        printf("Current Performance @K = %d: %.3f.\n", k, performance);
        if (performance > best_performance)
        {
            best_performance = performance;
            best_k = k;
        }
    }

    kmeans *km = new kmeans(best_k);
    km->set_training_data(dh->get_training_data());
    km->set_test_data(dh->get_test_data());
    km->set_validation_data(dh->get_validation_data());
    km->init_clusters();
    performance = km->test();

    printf("Current Performance @K = %d: %.3f.\n", best_k, performance);
}