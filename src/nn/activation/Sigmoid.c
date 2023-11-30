#include "matrix/Matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "nn/activation/Sigmoid.h"
#include "nn/Layer.h"


VAR __sigmoid(VAR x)
{
    return 1 / (1 + exp(-x));
}
void Inititalize_Sigmoid(Sigmoid* this, Matrix* input_matrix)
{
    Matrix* tmp_input = Init_Matrix(input_matrix->shape, input_matrix->size_shape, "null");
    this->input->Copy(this->input, tmp_input);
    tmp_input->Free(tmp_input);
}
Matrix* Forward_Sigmoid(Sigmoid* this, Matrix* input)
{
    Matrix* tmp = Init_Matrix(input->shape, input->size_shape, "null");
    VAR* array = input->array;
    for (int i = 0; i != input->size_array; i++)
    {   
        tmp->array[i] = __sigmoid(array[i]);
    }
    this->input->Copy(this->input, tmp);
    return tmp;


    // this->input->Copy(this->input, input);
    // return input->__Copy(*input);
}
Matrix* Backprop_Sigmoid (Sigmoid* this, Matrix* output, VAR learning_rate)
{
    learning_rate += 1;
    Matrix* input_gradient = Init_Matrix(this->input->shape, this->input->size_shape, "null");
    for (int i = 0; i != this->input->size_array; i++){
        input_gradient->array[i] = this->input->array[i] * (1 - this->input->array[i]);
    }
    input_gradient->Mult(input_gradient, output);
    return input_gradient;  
}
void Free_Sigmoid(Sigmoid* this){
    this->input->Free(this->input);
    free(this);
}
void Save_Sigmoid(Sigmoid* this, hid_t file_id)
{
    hid_t sigmoid_group = H5Gcreate(file_id, "sigmoid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    hsize_t dims[1] = {1};
    // Input shape
    dims[0] = this->input->size_shape; 
    hid_t shape_space = H5Screate_simple(1, dims, NULL);
    hid_t shape_dataset = H5Dcreate(sigmoid_group, "input_shape", H5T_NATIVE_INT, shape_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, this->input->shape);
    //Close
    H5Dclose(shape_dataset);
    H5Sclose(shape_space);

}
Layer* Load_Sigmoid(hid_t sigmoid_group)
{
    // Input shape
    Layer* sigmoid = Init_Sigmoid();
    Matrix* input = sigmoid->inheritance.sigmoid->input;

    hid_t shape_dataset = H5Dopen(sigmoid_group, "input_shape", H5P_DEFAULT);
    hid_t dataspace_size_shape = H5Dget_space(shape_dataset);
    hsize_t size_shape = H5Sget_simple_extent_npoints(dataspace_size_shape);
    int* input_shape = malloc(sizeof(int) * size_shape);
    H5Dread(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_shape);
    Matrix* tmp = Init_Matrix(input_shape, size_shape, "null");
    input->Copy(input, tmp);
    tmp->Free(tmp);
    H5Dclose(shape_dataset);
    H5Sclose(dataspace_size_shape);
    free(input_shape);
    return sigmoid;
}
Layer* Init_Sigmoid()
{
    Sigmoid* sigmoid = malloc(sizeof(Sigmoid));
    sigmoid->Initialize = Inititalize_Sigmoid;
    sigmoid->Backprop = Backprop_Sigmoid;
    sigmoid->Forward = Forward_Sigmoid;
    sigmoid->Save = Save_Sigmoid;
    int shape[2] = {1,1};
    sigmoid->input = Init_Matrix(shape, 2, "null");
    sigmoid->Free = Free_Sigmoid;
    sigmoid->output_size = 0;
    Layer* layer = Init_Layer();
    layer->inheritance.sigmoid = sigmoid;
    layer->layer = "sigmoid";
    return layer; 
}
