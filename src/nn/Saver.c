#include "nn/Saver.h"

void Save(Model* this, char* path)
{
    hid_t file_id = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    //Save the size of the shape
    hid_t dataspace_size_shape = H5Screate(H5S_SCALAR);
    hid_t dataset_size_shape = H5Dcreate(file_id, "size_shape", H5T_NATIVE_INT, dataspace_size_shape, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_size_shape, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->size_shape_input);

    //Save the shape
    hsize_t dims[1] = {2}; // Il y a deux éléments dans le tableau shape_data
    hid_t dataspace_shape = H5Screate_simple(1, dims, NULL);
    hid_t dataset_shape = H5Dcreate(file_id, "shape", H5T_NATIVE_INT, dataspace_shape, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_shape, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, this->shape_input);

    //Save the number of layers
    hid_t dataspace_nb_layers = H5Screate(H5S_SCALAR);
    hid_t dataset_nb_layers = H5Dcreate(file_id, "nb_layers", H5T_NATIVE_INT, dataspace_nb_layers, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_nb_layers, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->nb_layers);


    // Save the layers
    for (int i = 0; i != this->nb_layers; i++)
    {
        char name[] = "layer00";
        name[5] = i/10 + '0';
        name[6] = i%10 + '0';
        hid_t layer_name = H5Gcreate(file_id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        this->layers[i]->Save(this->layers[i], layer_name);
    }
    // Save loss
    size_t string_size = strlen(this->loss->name) + 1;
    hid_t string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, string_size);
    hid_t dataspace = H5Screate(H5S_SCALAR);
    hid_t dataset = H5Dcreate(file_id, "loss", string_type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, this->loss->name);

    //Save the learning rate
    hid_t dataspace_l_r = H5Screate(H5S_SCALAR);
    hid_t dataset_l_r = H5Dcreate(file_id, "learning_rate", H5T_NATIVE_VAR, dataspace_l_r, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_l_r, H5T_NATIVE_VAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->learning_rate);


    //Close the file
    H5Dclose(dataset_size_shape);
    H5Dclose(dataset_shape);
    H5Sclose(dataspace_size_shape);
    H5Sclose(dataspace_shape);
    H5Fclose(file_id);
    H5Dclose(dataset_nb_layers);
    H5Sclose(dataspace_nb_layers);
    H5Dclose(dataset_l_r);
    H5Sclose(dataspace_l_r);
    
}

Model* Load_Model(char* path)
{
    Optimizer* optimizer = Init_Adam(1e-3, 0.9, 0.999, 1e-8);
    hid_t file_id = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) 
    {
        fprintf(stderr, "Failed to open file %s\n", path);
        return NULL;
    }

    // Load the size of the shape
    int size_shape_input;
    hid_t dataset_size_shape = H5Dopen(file_id, "size_shape", H5P_DEFAULT);
    H5Dread(dataset_size_shape, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &size_shape_input);

    // Load the shape
    int shape_input[2];
    hid_t dataset_shape = H5Dopen(file_id, "shape", H5P_DEFAULT);
    H5Dread(dataset_shape, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shape_input);

    // Load the number of layers
    int nb_layers;
    hid_t dataset_nb_layers = H5Dopen(file_id, "nb_layers", H5P_DEFAULT);
    H5Dread(dataset_nb_layers, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nb_layers);

    // Load the layers
    Layer** layers = malloc(nb_layers * sizeof(Layer*));
    for (int i = 0; i != nb_layers; i++)
    {
        char name[] = "layer00";
        name[5] = i/10 + '0';
        name[6] = i%10 + '0';
        hid_t layer_name = H5Gopen(file_id, name, H5P_DEFAULT);
        layers[i] = Load_Layer(layer_name, optimizer);
    }

    // Load the loss
    hid_t dataset_loss = H5Dopen(file_id, "loss", H5P_DEFAULT);
    hid_t string_type = H5Dget_type(dataset_loss);
    size_t string_size = H5Tget_size(string_type);

    char* string_loss = malloc(string_size + 1);
    memset(string_loss, 0, string_size + 1);
    H5Dread(dataset_loss, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, string_loss);
    //load the learning rate
    VAR learning_rate;
    hid_t dataset_l_r = H5Dopen(file_id, "learning_rate", H5P_DEFAULT);
    H5Dread(dataset_l_r, H5T_NATIVE_VAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, &learning_rate);

    // Create the model
    //close 
    H5Dclose(dataset_size_shape);
    H5Dclose(dataset_shape);
    H5Dclose(dataset_nb_layers);
    H5Dclose(dataset_loss);
    H5Dclose(dataset_l_r);
    H5Fclose(file_id);

    Model* model = Init_Model(shape_input, size_shape_input);
    model->nb_layers = nb_layers;
    model->layers = layers;
    model->loss = Get_Loss(string_loss);
    model->learning_rate = learning_rate;
    model->optimizer = optimizer;
    model->is_compiled = 1;
    // model->Compile(model, Get_Loss(string_loss), learning_rate);
    free(string_loss);
    return model;
}
