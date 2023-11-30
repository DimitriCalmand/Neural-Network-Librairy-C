#include "matrix/Matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "nn/activation/Tanh.h"
#include "nn/Layer.h"

void Inititalize_Tanh(Tanh* this, Matrix* input_matrix)
{
    Matrix* tmp_input = Init_Matrix(input_matrix->shape, input_matrix->size_shape, "null");
    this->input->Copy(this->input, tmp_input);
    tmp_input->Free(tmp_input);
}
Matrix* Forward_Tanh(Tanh* this, Matrix* input)
{
    Matrix* tmp = Init_Matrix(input->shape, input->size_shape, "null");
    VAR* array = input->array;
    for (int i = 0; i != input->size_array; i++)
    {   
        tmp->array[i] = tanh(array[i]);
    }
    this->input->Copy(this->input, tmp);
    return tmp;
}
Matrix* Backprop_Tanh (Tanh* this, Matrix* output, VAR learning_rate){
    learning_rate += 1;
    Matrix* input_gradient = Init_Matrix(this->input->shape, this->input->size_shape, "null");
    for (int i = 0; i != this->input->size_array; i++)
    {
        VAR value = this->input->array[i];
        input_gradient->array[i] = value * (1 - value*value);
    }
    input_gradient->Mult(input_gradient, output);
    return input_gradient;  
}
void Free_Tanh(Tanh* this){
    this->input->Free(this->input);
    free(this);
}
Layer* Init_Tanh(){
    Tanh* tanh = malloc(sizeof(Tanh));
    tanh->Initialize = Inititalize_Tanh;
    tanh->Backprop = Backprop_Tanh;
    tanh->Forward = Forward_Tanh;
    int shape[2] = {1,1};
    tanh->input = Init_Matrix(shape, 2, "null");
    tanh->Free = Free_Tanh;
    tanh->output_size = 0;
    Layer* layer = Init_Layer();
    layer->inheritance.tanh = tanh;
    layer->layer = "tanh";
    return layer; 
}
