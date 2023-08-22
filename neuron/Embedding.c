#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Embedding.h"
#include <string.h>

void Set_Optimizer_Embedding(Embedding* this, Optimizer* optimizer)
{
    this->Optimizer_weights = optimizer->Create_Optimizer(optimizer, this->weights);
}
void Initialize_Embedding(Embedding* this, Matrix* input_matrix, Optimizer* optimizer)
{
    Matrix* tmp = Init_Matrix((int[]){this->vocab_size, this->embedding_dim}, 2, "random");
    this->weights->Copy(this->weights, tmp);
    tmp->Free(tmp);
    this->input->Copy(this->input, input_matrix);
    Set_Optimizer_Embedding(this, optimizer);
}
Matrix* Forward_Embedding(Embedding* this, Matrix* input)
{
    this->input->Copy(this->input, input);
    int shape[] = {input->shape[0], input->shape[1], this->embedding_dim};
    Matrix* res = Init_Matrix(shape, 3, "null");
    // printf("maxim %f\n", input->Max(input));
    for (int i = 0; i!= shape[0]; i++)
    {
        for (int j = 0; j!= shape[1]; j++)
        {
            int index = i*shape[1]+j;
            int res_index = i*shape[1]*shape[2] + j*shape[2];
            int index_weight = input->array[index]*this->embedding_dim;
            VAR* tmp = this->weights->array+index_weight;
            memcpy(res->array+res_index, tmp, sizeof(VAR)*this->embedding_dim);
        }
    }
    // printf("end\n");

    return res;
}
Matrix* Backprop_Embedding(Embedding* this, Matrix* output)
{
    Matrix* res = Init_Matrix(this->input->shape, 2, "null");
    int shape[] = {output->shape[0], output->shape[1], this->embedding_dim};
    Matrix* grad_weights = Init_Matrix(this->weights->shape, 2, "null");
    // Calculate gradients
    for (int i = 0; i < shape[0]; i++)
    {
        for (int j = 0; j < shape[1]; j++)
        {
            int input_index = i * this->input->shape[1] + j;
            int output_index = i * shape[1] * shape[2] + j * shape[2];

            int weight_index = (int)this->input->array[input_index] * this->embedding_dim;

            for (int k = 0; k < this->embedding_dim; k++)
            {
                grad_weights->array[weight_index + k] += output->array[output_index + k];
            }
        }
    }
    // Update weights with gradient and learning rate
    this->Optimizer_weights->Optimize(this->Optimizer_weights, grad_weights);
    // grad_weights->Mult_Scalar(grad_weights, learning_rate);
    // this->weights->Sub(this->weights, grad_weights);
    
    // Free the allocated memory for grad_weights
    grad_weights->Free(grad_weights);
    return res; //Return nothing because embedding is the first layer
}

void Free_Embedding(Embedding* this)
{
    this->weights->Free(this->weights);
    this->input->Free(this->input);
    this->Optimizer_weights->Free(this->Optimizer_weights);
    free(this);
}
void Save_Embedding(Embedding* this,  hid_t file_id)
{
    hid_t embedding_group = H5Gcreate(file_id, "embedding", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // ouput_size
    hsize_t dims[1] = {1};
    hid_t dataspace_vocab = H5Screate_simple(1, dims, NULL);
    hid_t dataset_vocab = H5Dcreate(embedding_group, "vocab_size", H5T_NATIVE_INT, dataspace_vocab, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_vocab, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(this->vocab_size));
    H5Dclose(dataset_vocab);
    H5Sclose(dataspace_vocab);

    hid_t dataspace_embedding_dim = H5Screate_simple(1, dims, NULL);
    hid_t dataset_embedding_dim = H5Dcreate(embedding_group, "embedding_dim", H5T_NATIVE_INT, dataspace_embedding_dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_embedding_dim, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(this->embedding_dim));
    H5Dclose(dataset_embedding_dim);
    H5Sclose(dataspace_embedding_dim);

    dims[0] = this->input->size_shape; 
    hid_t shape_space = H5Screate_simple(1, dims, NULL);
    hid_t shape_dataset = H5Dcreate(embedding_group, "input_shape", H5T_NATIVE_INT, shape_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, this->input->shape);
    //Close
    H5Dclose(shape_dataset);
    H5Sclose(shape_space);
    // weights
    this->weights->Save(this->weights, embedding_group, "weights");
}
Layer* Load_Embedding(hid_t file, Optimizer* optimizer)
{
    Embedding* embedding = malloc(sizeof(Embedding));

    hid_t data_vocab = H5Dopen(file, "vocab_size", H5P_DEFAULT);
    H5Dread(data_vocab,  H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(embedding->vocab_size));
    H5Dclose(data_vocab);
    hid_t data_embedding = H5Dopen(file, "embedding_dim", H5P_DEFAULT);
    H5Dread(data_embedding,  H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(embedding->embedding_dim));
    H5Dclose(data_embedding);

    hid_t shape_dataset = H5Dopen(file, "input_shape", H5P_DEFAULT);
    hid_t dataspace_size_shape = H5Dget_space(shape_dataset);
    hsize_t size_shape = H5Sget_simple_extent_npoints(dataspace_size_shape);
    int* input_shape = malloc(sizeof(int) * size_shape);
    H5Dread(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_shape);
    embedding->input = Init_Matrix(input_shape, size_shape, "null");
    H5Dclose(shape_dataset);
    H5Sclose(dataspace_size_shape);

    embedding->weights = Load_Matrix(file, "weights");
    Layer* layer = Init_Layer();
    layer->inheritance.embedding = embedding;
    layer->layer = "embedding";
    Set_Optimizer_Embedding(embedding, optimizer);
    embedding->Forward = Forward_Embedding;
    embedding->Backprop = Backprop_Embedding;
    embedding->Initialize = Initialize_Embedding;
    embedding->Free = Free_Embedding;
    embedding->Save = Save_Embedding;

    free(input_shape);
    return layer;

}

Layer* Init_Embedding(int vocab_size, int embedding_dim){
    
    Embedding* embedding = malloc(sizeof(Embedding));
    embedding->vocab_size = vocab_size;
    embedding->embedding_dim = embedding_dim;
    embedding->Forward = Forward_Embedding;
    embedding->Backprop = Backprop_Embedding;
    embedding->Initialize = Initialize_Embedding;
    embedding->Free = Free_Embedding;
    embedding->Save = Save_Embedding;
    embedding->weights = Init_Matrix((int[]){1,}, 1, "null");
    embedding->input = Init_Matrix((int[]){1,}, 1, "null");
    Layer* layer = Init_Layer();
    layer->inheritance.embedding = embedding;
    layer->layer = "embedding";

    return layer;
}