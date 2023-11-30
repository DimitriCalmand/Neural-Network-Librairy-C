#include <string.h>
#include "matrix/Matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "nn/Dense.h"
#include "nn/activation/Sigmoid.h"
#include "nn/Layer.h"

void Initialize_Layer(Layer* this, Matrix* input_matrix, Optimizer* optimizer)
{
    char* layer = this->layer;
    if (strcmp(layer, "sigmoid") == 0)
    {
        Sigmoid* sigmoid = this->inheritance.sigmoid;
        sigmoid->Initialize(sigmoid, input_matrix);
    }
    else if (strcmp(layer, "dense") == 0)
    {
        Dense* dense = this->inheritance.dense;
        dense->Initialize(dense, input_matrix, optimizer);
    }
    else if (strcmp(layer, "tanh") == 0)
    {
        Tanh* tanh = this->inheritance.tanh;
        tanh->Initialize(tanh, input_matrix);
    }
    else if (strcmp(layer, "relu") == 0)
    {
        Relu* relu = this->inheritance.relu;
        relu->Initialize(relu, input_matrix);
    }
    else if (strcmp(layer, "softmax") == 0)
    {
        Softmax* softmax = this->inheritance.softmax;
        softmax->Initialize(softmax, input_matrix);
    }
    else if (strcmp(layer, "leaky_relu") == 0)
    {
        Leaky_Relu* leaky_relu = this->inheritance.leaky_relu;
        leaky_relu->Initialize(leaky_relu, input_matrix);
    }
    else if (strcmp(layer, "lstm") == 0)
    {
        Lstm* lstm = this->inheritance.lstm;
        lstm->Initialize(lstm, input_matrix, optimizer);
    }
    else if (strcmp(layer, "embedding") == 0)
    {
        Embedding* embedding = this->inheritance.embedding;
        embedding->Initialize(embedding, input_matrix, optimizer);
    }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name : Layer.c");
    }

}
Matrix* Forward_Layer(Layer* this, Matrix* input)
{
    char* layer = this->layer;
    if (strcmp(layer, "sigmoid") == 0)
    {
        Sigmoid* sigmoid = this->inheritance.sigmoid;
        return sigmoid->Forward(sigmoid, input);
    }
    else if (strcmp(layer, "dense") == 0)
    {
        Dense* dense = this->inheritance.dense;
        return dense->Forward(dense, input);
    }
    else if (strcmp(layer, "tanh") == 0)
    {
        Tanh* tanh = this->inheritance.tanh;
        return tanh->Forward(tanh, input);
    }
    else if (strcmp(layer, "relu") == 0)
    {
        Relu* relu = this->inheritance.relu;
        return relu->Forward(relu, input);
    }
    else if (strcmp(layer, "softmax") == 0)
    {
        Softmax* softmax = this->inheritance.softmax;
        return softmax->Forward(softmax, input);
    }
    else if (strcmp(layer, "leaky_relu") == 0)
    {
        Leaky_Relu* leaky_relu = this->inheritance.leaky_relu;
        return leaky_relu->Forward(leaky_relu, input);
    }
    else if (strcmp(layer, "lstm") == 0)
    {
        Lstm* lstm = this->inheritance.lstm;
        return lstm->Forward(lstm, input);
    }
    else if (strcmp(layer, "embedding") == 0)
    {
        Embedding* embedding = this->inheritance.embedding;
        return embedding->Forward(embedding, input);
    }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name : Layer.c");
    }
}
Matrix* Backprop_Layer(Layer* this, Matrix* output, VAR learning_rate)
{
    char* layer = this->layer;
    if (strcmp(layer, "sigmoid") == 0)
    {
        Sigmoid* sigmoid = this->inheritance.sigmoid;
        return sigmoid->Backprop(sigmoid, output, learning_rate);
    }
    else if (strcmp(layer, "dense") == 0)
    {
        Dense* dense = this->inheritance.dense;
        return dense->Backprop(dense, output);    
    }
    else if (strcmp(layer, "tanh") == 0)
    {
        Tanh* tanh = this->inheritance.tanh;
        return tanh->Backprop(tanh, output, learning_rate);
    }
    else if (strcmp(layer, "relu") == 0)
    {
        Relu* relu = this->inheritance.relu;
        return relu->Backprop(relu, output, learning_rate);
    }
    else if (strcmp(layer, "softmax") == 0)
    {
        Softmax* softmax = this->inheritance.softmax;
        return softmax->Backprop(softmax, output, learning_rate);
    }
    else if (strcmp(layer, "leaky_relu") == 0)
    {
        Leaky_Relu* leaky_relu = this->inheritance.leaky_relu;
        return leaky_relu->Backprop(leaky_relu, output, learning_rate);
    }
    else if (strcmp(layer, "lstm") == 0)
    {
        Lstm* lstm = this->inheritance.lstm;
        return lstm->Backprop(lstm, output);
    }
    else if (strcmp(layer, "embedding") == 0)
    {
        Embedding* embedding = this->inheritance.embedding;
        return embedding->Backprop(embedding, output);
    }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name : Layer.c");
    }
}
void Free_Layer(Layer* this)
{
    char* layer = this->layer;
    if (strcmp(layer, "sigmoid") == 0)
    {
        Sigmoid* sigmoid = this->inheritance.sigmoid;
        sigmoid->Free(sigmoid);
    }
    else if(strcmp(layer, "dense") == 0)
    {
        Dense* dense = this->inheritance.dense;
        dense->Free(dense);
    }
    else if (strcmp(layer, "tanh") == 0)
    {
        Tanh* tanh = this->inheritance.tanh;
        tanh->Free(tanh);
    }
    else if (strcmp(layer, "relu") == 0)
    {
        Relu* relu = this->inheritance.relu;
        relu->Free(relu);
    }
    else if (strcmp(layer, "softmax") == 0)
    {
        Softmax* softmax = this->inheritance.softmax;
        softmax->Free(softmax);
    }
    else if (strcmp(layer, "leaky_relu") == 0)
    {
        Leaky_Relu* leaky_relu = this->inheritance.leaky_relu;
        leaky_relu->Free(leaky_relu);
    }
    else if (strcmp(layer, "lstm") == 0)
    {
        Lstm* lstm = this->inheritance.lstm;
        lstm->Free(lstm);
    }
    else if (strcmp(layer, "embedding") == 0)
    {
        Embedding* embedding = this->inheritance.embedding;
        embedding->Free(embedding);
    }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name : Layer.c");
    }
    free(this);
}
void Save_layer(Layer* this,  hid_t file_id)
{
    char* layer = this->layer;
    if(strcmp(layer, "dense") == 0)
    {
        Dense* dense = this->inheritance.dense;
        dense->Save(dense, file_id);
    }
    else if (strcmp(layer, "lstm") == 0)
    {
        Lstm* lstm = this->inheritance.lstm;
        lstm->Save(lstm, file_id);
    }
    else if (strcmp(layer, "embedding") == 0)
    {
        Embedding* embedding = this->inheritance.embedding;
        embedding->Save(embedding, file_id);
    }
    else if (strcmp(layer, "sigmoid") == 0)
    {
        Sigmoid* sigmoid = this->inheritance.sigmoid;
        sigmoid->Save(sigmoid, file_id);
    }
    // else if (strcmp(layer, "tanh") == 0)
    // {
    //     Tanh* tanh = this->inheritance.tanh;
    //     tanh->Save(tanh, file_id);
    // }
    // else if (strcmp(layer, "relu") == 0)
    // {
    //     Relu* relu = this->inheritance.relu;
    //     relu->Save(relu, file_id);
    // }
    // else if (strcmp(layer, "softmax") == 0)
    // {
    //     Softmax* softmax = this->inheritance.softmax;
    //     softmax->Save(softmax, file_id);
    // }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name : Layer.c");
    }
}
Layer* Load_Layer(hid_t layer_group, Optimizer* optimizer)
{
    char layer_type[64];
    H5L_info_t link_info;
    H5Lget_info_by_idx(layer_group, ".", H5_INDEX_NAME, H5_ITER_INC, 0, &link_info, H5P_DEFAULT);
    H5Lget_name_by_idx(layer_group, ".", H5_INDEX_NAME, H5_ITER_INC, 0, layer_type, 64, H5P_DEFAULT);
    if (strcmp(layer_type, "dense") == 0) 
    {
        hid_t dense_group = H5Gopen(layer_group, "dense", H5P_DEFAULT);
        return Load_Dense(dense_group, optimizer);
    }
    else if (strcmp(layer_type, "sigmoid") == 0)
    {
        hid_t sigmoid_group = H5Gopen(layer_group, "sigmoid", H5P_DEFAULT);
        return Load_Sigmoid(sigmoid_group);
    }
    else if(strcmp(layer_type, "lstm") == 0)
    {
        return Load_Lstm(layer_group, optimizer);
    }
    else if(strcmp(layer_type, "embedding") == 0)
    {
        hid_t embedding_group = H5Gopen(layer_group, "embedding", H5P_DEFAULT);
        return Load_Embedding(embedding_group, optimizer);
    }
    else 
    {
        printf("Unknown layer type: %s\n", layer_type);
    }
    return NULL;
}
Matrix* Get_Input(Layer* this)
{
    char* layer = this->layer;
    if (strcmp(layer, "sigmoid") == 0)
    {
        Sigmoid* sigmoid = this->inheritance.sigmoid;
        return sigmoid->input;
    }
    else if(strcmp(layer, "dense") == 0)
    {
        Dense* dense = this->inheritance.dense;
        return dense->input;
    }
    else if (strcmp(layer, "tanh") == 0)
    {
        Tanh* tanh = this->inheritance.tanh;
        return tanh->input;
    }
    else if (strcmp(layer, "relu") == 0)
    {
        Relu* relu = this->inheritance.relu;
        return relu->input;
    }
    else if (strcmp(layer, "softmax") == 0)
    {
        Softmax* softmax = this->inheritance.softmax;
        return softmax->input;
    }
    else if (strcmp(layer, "leaky_relu") == 0)
    {
        Leaky_Relu* leaky_relu = this->inheritance.leaky_relu;
        return leaky_relu->input;
    }
    else if (strcmp(layer, "lstm") == 0)
    {
        Lstm* lstm = this->inheritance.lstm;
        return lstm->input;
    }
    else if (strcmp(layer, "embedding") == 0)
    {
        Embedding* embedding = this->inheritance.embedding;
        return embedding->input;
    }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name : Layer.c");
    }
}
Layer* Init_Layer()
{
    Layer* layer = malloc(sizeof(Layer));
    layer->Backprop = Backprop_Layer;
    layer->Free = Free_Layer;
    layer->Forward = Forward_Layer;
    layer->Initialize = Initialize_Layer;
    layer->Save = Save_layer;
    layer->Get_Input = Get_Input;
    return layer;
}
