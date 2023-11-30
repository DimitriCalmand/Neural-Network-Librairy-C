#include "matrix/Matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "nn/activation/Relu.h"
#include "nn/Layer.h"

void Inititalize_Relu(Relu* this, Matrix* input_matrix)
{
    Matrix* tmp_input = Init_Matrix(input_matrix->shape, input_matrix->size_shape, "null");
    this->input->Copy(this->input, tmp_input);
    tmp_input->Free(tmp_input);
}
VAR __max(VAR a, VAR b){
    if (a > b)
        return a;
    return b;
}
Matrix* Forward_Relu(Relu* this, Matrix* input)
{
    Matrix* tmp = Init_Matrix(input->shape, input->size_shape, "null");
    VAR* array = input->array;
    for (int i = 0; i != input->size_array; i++)
    {   
        tmp->array[i] =  __max(array[i], 0);
    }
    this->input->Copy(this->input, tmp);
    return tmp;
}
Matrix* Backprop_Relu (Relu* this, Matrix* output, VAR learning_rate){
    learning_rate += 1;
    Matrix* input_gradient = Init_Matrix(this->input->shape, this->input->size_shape, "null");
    for (int i = 0; i != this->input->size_array; i++)
    {
        if (this->input->array[i] > 0)
        {
            input_gradient->array[i] = 1;
        }
        else
        {
            input_gradient->array[i] = 0;
        }
    }
    input_gradient->Mult(input_gradient, output);
    return input_gradient;  
}
void Free_Relu(Relu* this)
{
    this->input->Free(this->input);
    free(this);
}
Layer* Init_Relu()
{
    Relu* relu = malloc(sizeof(Relu));
    relu->Initialize = Inititalize_Relu;
    relu->Backprop = Backprop_Relu;
    relu->Forward = Forward_Relu;
    int shape[2] = {1,1};
    relu->input = Init_Matrix(shape, 2, "null");
    relu->Free = Free_Relu;
    relu->output_size = 0;
    Layer* layer = Init_Layer();
    layer->inheritance.relu = relu;
    layer->layer = "relu";
    return layer; 
}
