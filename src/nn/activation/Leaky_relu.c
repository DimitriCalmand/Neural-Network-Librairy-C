#include "matrix/Matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "nn/activation/Leaky_relu.h"
#include "nn/Layer.h"

VAR __leaky_relu(VAR x)
{
    if (x > 0)
        return x;
    else
        return 0.01 * x;
}
void Inititalize_Leaky_Relu(Leaky_Relu* this, Matrix* input_matrix)
{
    Matrix* tmp_input = Init_Matrix(input_matrix->shape, input_matrix->size_shape, "null");
    this->input->Copy(this->input, tmp_input);
    tmp_input->Free(tmp_input);
}
Matrix* Forward_Leaky_Relu(Leaky_Relu* this, Matrix* input)
{
    Matrix* tmp = Init_Matrix(input->shape, input->size_shape, "null");
    VAR* array = input->array;
    for (int i = 0; i != input->size_array; i++)
    {   
        tmp->array[i] = __leaky_relu(array[i]);
    }
    this->input->Copy(this->input, tmp);
    return tmp;
}
Matrix* Backprop_Leaky_Relu (Leaky_Relu* this, Matrix* output, VAR learning_rate)
{
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
            input_gradient->array[i] = 0.01;
        }
    }
    input_gradient->Mult(input_gradient, output);
    return input_gradient;  
}
void Free_Leaky_Relu(Leaky_Relu* this){
    this->input->Free(this->input);
    free(this);
}
Layer* Init_Leaky_Relu(){
    Leaky_Relu* leaky_relu = malloc(sizeof(Leaky_Relu));
    leaky_relu->Initialize = Inititalize_Leaky_Relu;
    leaky_relu->Backprop = Backprop_Leaky_Relu;
    leaky_relu->Forward = Forward_Leaky_Relu;
    int shape[2] = {1,1};
    leaky_relu->input = Init_Matrix(shape, 2, "null");
    leaky_relu->Free = Free_Leaky_Relu;
    leaky_relu->output_size = 0;
    Layer* layer = Init_Layer();
    layer->inheritance.leaky_relu = leaky_relu;
    layer->layer = "leaky_relu";
    return layer; 
}
