#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn/Dense.h"
#include <string.h>

void Set_Optimizer_Dense(Dense* this, Optimizer* Optimizer)
{
    this->Optimizer_weights = Optimizer->Create_Optimizer(Optimizer, this->weights);
    this->Optimizer_bias = Optimizer->Create_Optimizer(Optimizer, this->bias);
}
void Initialize_Dense(Dense* this, Matrix* input_matrix, Optimizer* Optimizer)
{
    int shape[2] = {input_matrix->shape[input_matrix->size_shape-1], this->output_size};
    Matrix* tmp_weight = Init_Matrix(shape ,2,"random");
    this->weights->Copy(this->weights, tmp_weight);
    tmp_weight->Free(tmp_weight);
    shape[0] = 1;
    Matrix* tmp_bias = Init_Matrix(shape,2,"0");
    this->bias->Copy(this->bias, tmp_bias);
    tmp_bias->Free(tmp_bias);
    this->input->Copy(this->input, input_matrix);
    Set_Optimizer_Dense(this, Optimizer);
}

Matrix* Forward_Dense(Dense* this, Matrix* input)
{
    this->input->Copy(this->input, input);
    Matrix* tmp_dot = input->__Dot(*input, *(this->weights));
    tmp_dot->Add(tmp_dot, this->bias);
    return tmp_dot;
}

Matrix* Backprop_Dense(Dense* this, Matrix* output)
{
    if (this->input->size_shape == 3)
    {
        Matrix* gradient_w = Init_Matrix(this->weights->shape, this->weights->size_shape, "0");
        Matrix* gradient_b = Init_Matrix(this->bias->shape, this->bias->size_shape, "0");
        Matrix* res = Init_Matrix(this->input->shape, this->input->size_shape, "null");
        for (int t = this->input->shape[1]-1; t>-1; t--)
        {
            Matrix* output_t = output->__Slicing(*output, 1, t, (t+1));            
            Matrix* input_t_1 = this->input->__Slicing(*(this->input), 1, t, (t+1));
            input_t_1->Transpose(input_t_1, -1, -2);
            Matrix* tmp = this->weights->__Transpose(*(this->weights),-1,-2);
            Matrix* tmp_res = output_t->__Dot(*output_t, *tmp);
            input_t_1->Dot(input_t_1, output_t);
            gradient_w->Add(gradient_w, input_t_1);
            Matrix* tmp_cols_sum = output_t->__Sum(*output_t, -2);
            tmp_cols_sum->Expand_Dim(tmp_cols_sum, 0);
            // tmp_cols_sum->Mult_Scalar(tmp_cols_sum, learning_rate);
            gradient_b->Add(gradient_b, tmp_cols_sum);
            tmp_cols_sum->Free(tmp_cols_sum);
            input_t_1->Free(input_t_1);
            tmp->Free(tmp);
            output_t->Free(output_t);
            res->Put(res, tmp_res, 1, t, (t+1));
            tmp_res->Free(tmp_res);
        }
        this->Optimizer_weights->Optimize(this->Optimizer_weights, gradient_w);
        this->Optimizer_bias->Optimize(this->Optimizer_bias, gradient_b);
        gradient_w->Free(gradient_w);
        gradient_b->Free(gradient_b);
        return res;
    }

    Matrix* tmp_dot = this->weights->__Transpose(*(this->weights),-1,-2);
    Matrix* res = tmp_dot->__Dot(*output, *tmp_dot);
    tmp_dot->Free(tmp_dot);
    /*      Update the weights      */ 
    Matrix* tmp_transpose = this->input->__Transpose(*(this->input), -1,-2);
    tmp_transpose->Dot(tmp_transpose, output);

    this->Optimizer_weights->Optimize(this->Optimizer_weights, tmp_transpose);
    tmp_transpose->Free(tmp_transpose);

    Matrix* tmp_cols_sum = output->__Sum(*output, -2);

    tmp_cols_sum->Expand_Dim(tmp_cols_sum, 0);

    this->Optimizer_bias->Optimize(this->Optimizer_bias, tmp_cols_sum);
    tmp_cols_sum->Free(tmp_cols_sum);

    return res;
}
void Free_Dense(Dense* this){
    this->bias->Free(this->bias);
    this->weights->Free(this->weights);
    this->input->Free(this->input);
    this->Optimizer_bias->Free(this->Optimizer_bias);
    this->Optimizer_weights->Free(this->Optimizer_weights);
    free(this);
}

void Save_Dense(Dense* this, hid_t file_id)
{
    hid_t Dense_group = H5Gcreate(file_id, "dense", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // ouput_size
    hsize_t dims[1] = {1};
    hid_t dataspace = H5Screate_simple(1, dims, NULL);
    hid_t dataset = H5Dcreate(Dense_group, "output_size", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(this->output_size));
    H5Dclose(dataset);
    H5Sclose(dataspace);

    // Input shape
    dims[0] = this->input->size_shape; 
    hid_t shape_space = H5Screate_simple(1, dims, NULL);
    hid_t shape_dataset = H5Dcreate(Dense_group, "input_shape", H5T_NATIVE_INT, shape_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, this->input->shape);
    //Close
    H5Dclose(shape_dataset);
    H5Sclose(shape_space);
    // weights
    this->weights->Save(this->weights, Dense_group, "weights");
    // bias
    this->bias->Save(this->bias, Dense_group, "bias");
    // close
    H5Gclose(Dense_group);
}
void Function_Dense(Dense* dense)
{
    dense->Forward = Forward_Dense;
    dense->Backprop = Backprop_Dense;
    dense->Initialize = Initialize_Dense;
    dense->Free = Free_Dense;
    dense->Save = Save_Dense;
}
Layer* Load_Dense(hid_t file_id, Optimizer* optimizer)
{
    Dense* dense = malloc(sizeof(Dense));
    Function_Dense(dense);
    // output_size
    hid_t dataset = H5Dopen(file_id, "output_size", H5P_DEFAULT);
    H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(dense->output_size));
    H5Dclose(dataset);

    //input shape
    hid_t shape_dataset = H5Dopen(file_id, "input_shape", H5P_DEFAULT);
    hid_t dataspace_size_shape = H5Dget_space(shape_dataset);
    hsize_t size_shape = H5Sget_simple_extent_npoints(dataspace_size_shape);
    int* input_shape = malloc(sizeof(int) * size_shape);
    H5Dread(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_shape);
    dense->input = Init_Matrix(input_shape, size_shape, "null");
    H5Dclose(shape_dataset);
    H5Sclose(dataspace_size_shape);
    // weights
    dense->weights = Load_Matrix(file_id, "weights");
    // bias
    dense->bias = Load_Matrix(file_id, "bias");
    Set_Optimizer_Dense(dense, optimizer);
    Layer* layer = Init_Layer();
    layer->inheritance.dense = dense;
    layer->layer = "dense";


    free(input_shape);


    
    return layer;
}

Layer* Init_Dense(int output_size){
    Dense* dense = malloc(sizeof(Dense));

    Function_Dense(dense);
    int shape[2] = {1,1};
    dense->weights = Init_Matrix(shape,0,"null");
    dense->bias = Init_Matrix(shape,0,"null");
    dense->input = Init_Matrix(shape,0,"null");
    dense->output_size = output_size;
    Layer* layer = Init_Layer();
    layer->inheritance.dense = dense;
    layer->layer = "dense";
    
    return layer;
}
