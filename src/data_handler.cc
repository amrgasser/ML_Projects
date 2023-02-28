#include "data_handler.hpp"

data_handler::data_handler()
{
    data_array = new std::vector<data *>;
    test_data = new std::vector<data *>;
    training_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
}
data_handler::~data_handler()
{
    // FREE dynamically allocated stuff
}

void data_handler::read_feature_vector(std::string path)
{
    uint32_t header[4]; // | MAGIC | NUM IMAGES | ROWSIZE | COLSIZE |
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f)
    {
        for (int i = 0; i < 4; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting file header.\n");
        int image_size = header[2] * header[3];
        for (int i = 0; i < header[1]; ++i)
        {
            data *d = new data();
            uint8_t element[1];
            for (int j = 0; j < image_size; ++j)
            {
                if (fread(element, sizeof(element), 1, f))
                {
                    d->append_to_feature_vector(element[0]);
                }
                else
                {
                    printf("Error reading from file.\n");
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
        printf("Data read and stored num: %lu.\n", data_array->size());
    }
    else
    {
        printf("File not fouund.\n");
        exit(1);
    }
}
void data_handler::read_feature_labels(std::string path)
{
    uint32_t header[2]; // | MAGIC | NUM LABELS |
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f)
    {
        for (int i = 0; i < 2; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting Label file header.\n");
        for (int i = 0; i < header[1]; ++i)
        {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f))
            {
                data_array->at(i)->set_label(element[0]);
            }
            else
            {
                printf("Error reading from labels file.\n");
                exit(1);
            }
        }
        printf("Labels read and stored.\n");
    }
    else
    {
        printf("Labels File not found.\n");
        exit(1);
    }
}
void data_handler::split_data()
{
    std::unordered_set<int> used_indexes;
    int train_size = data_array->size() * TRAINING_SET_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;
    int valid_size = data_array->size() * VALIDATION_SET_PERCENT;
    int count = 0;
    while (count < train_size)
    {
        int rand_ind = rand() % data_array->size();
        if (used_indexes.find(rand_ind) == used_indexes.end())
        {
            training_data->push_back(data_array->at(rand_ind));
            used_indexes.insert(rand_ind);
            count++;
        }
    }
    printf("Training Data size: %d.\n", train_size);
    count = 0;
    while (count < test_size)
    {
        int rand_ind = rand() % data_array->size();
        if (used_indexes.find(rand_ind) == used_indexes.end())
        {
            test_data->push_back(data_array->at(rand_ind));
            used_indexes.insert(rand_ind);
            count++;
        }
    }
    printf("Test Data size: %d.\n", test_size);

    count = 0;

    while (count < valid_size)
    {
        int rand_ind = rand() % data_array->size();
        if (used_indexes.find(rand_ind) == used_indexes.end())
        {
            validation_data->push_back(data_array->at(rand_ind));
            used_indexes.insert(rand_ind);
            count++;
        }
    }
    printf("Validation Data size: %d.\n", test_size);
}
void data_handler::count_classes()
{
    int count = 0;
    for (unsigned i = 0; i < data_array->size(); i++)
    {
        if (class_map.find(data_array->at(i)->get_label()) == class_map.end())
        {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enum_label(count);
            count++;
        }
    }
    num_classes = count;
    for (data *data : *data_array)
        data->set_class_vector(num_classes);
    printf("Classes found: %d.\n", count);
}

uint32_t data_handler::convert_to_little_endian(const unsigned char *bytes)
{
    return (uint32_t)((bytes[0] << 24) |
                      (bytes[1] << 16) |
                      (bytes[2] << 8) |
                      (bytes[3]));
}

int data_handler::get_class_counts()
{
    return num_classes;
}

std::vector<data *> *data_handler::get_training_data()
{
    return training_data;
}
std::vector<data *> *data_handler::get_test_data()
{
    return test_data;
}
std::vector<data *> *data_handler::get_validation_data()
{
    return validation_data;
}

void data_handler::read_csv(std::string path, std::string delimiter)
{
    num_classes = 0;
    std::ifstream data_file(path.c_str());
    std::string line;

    while (std::getline(data_file, line))
    {
        if (line.length() == 0)
            continue;
        data *d = new data();
        d->set_feature_vector(new std::vector<double>());
        size_t position = 0;
        std::string token; // value in between delimiter

        while ((position = line.find(delimiter) != std::string::npos))
        {
            token = line.substr(0, position);
            d->append_to_feature_vector(std::stod(token));
            line.erase(0, position + delimiter.length());
        }
        if (classMap.find(line) != classMap.end())
        {
            d->set_label(classMap[line]);
        }
        else
        {
            classMap[line] = num_classes;
            d->set_label(classMap[line]);
            num_classes++;
        }
        data_array->push_back(d);
    }
    feature_vector_size = data_array->at(0)->get_double_feature_vector()->size();
}

// int main()
// {
//     data_handler *dh = new data_handler();
//     dh->read_feature_vector("./train-images-idx3-ubyte");
//     dh->read_feature_labels("./train-labels-idx1-ubyte");
//     dh->split_data();
//     dh->count_classes();
// }