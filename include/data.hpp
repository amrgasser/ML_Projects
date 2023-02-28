#ifndef __DATA_H
#define __DATA_H
#include <set>
#include "stdint.h"
#include "stdio.h"

class data
{
    std::vector<uint8_t> *feature_vector;

    // for iris_dataset
    std::vector<double> *double_feature_vector;
    std::vector<int> *class_vector;

    uint8_t label;
    int enum_label;
    double distance;

public:
    data();
    ~data();
    void append_to_feature_vector(uint8_t);
    void append_to_feature_vector(double);

    void set_feature_vector(std::vector<uint8_t> *);
    void set_feature_vector(std::vector<double> *);
    void set_class_vector(int);
    void set_label(uint8_t);
    void set_enum_label(int);
    void set_distance(double val);

    uint8_t get_label();
    int get_feature_vector_size();
    int get_enum_label();
    double get_distance();
    std::vector<uint8_t> *get_feature_vector();
    std::vector<double> *get_double_feature_vector();
    std::vector<int> *get_class_vector();
};

#endif
