#include "matrix/Matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "nn/activation/Softmax.h"
#include "nn/Layer.h"


void Inititalize_Softmax(Softmax* this, Matrix* input_matrix)
{
    Matrix* tmp_input = Init_Matrix(input_matrix->shape, input_matrix->size_shape, "null");
    this->input->Copy(this->input, tmp_input);
    tmp_input->Free(tmp_input);
}
Matrix* Forward_Softmax(Softmax* this, Matrix* input)
{
    Matrix* exp_matrix = input->__Exponential(*input);
    Matrix* exp_sum = input->__Sum(*exp_matrix, -1);
    exp_sum->Expand_Dim(exp_sum, 1);
    Matrix* exp_matrix_div = exp_matrix->__Div(*exp_matrix, *exp_sum);
    this->input->Copy(this->input, exp_matrix_div);
    exp_matrix->Free(exp_matrix);
    exp_sum->Free(exp_sum);
    return exp_matrix_div;
}
Matrix* Backprop_Softmax (Softmax* this, Matrix* output, VAR learning_rate)
{
    learning_rate += 1;
    Matrix* input_gradient = Init_Matrix(this->input->shape, this->input->size_shape, "null");
    for (int i = 0; i != this->input->size_array; i++)
    {
        input_gradient->array[i] = this->input->array[i] * (1 - this->input->array[i]);
    }
    input_gradient->Mult(input_gradient, output);
    return input_gradient;  
}
void Free_Softmax(Softmax* this)
{
    this->input->Free(this->input);
    free(this);
}
Layer* Init_Softmax()
{
    Softmax* softmax = malloc(sizeof(Softmax));
    softmax->Initialize = Inititalize_Softmax;
    softmax->Backprop = Backprop_Softmax;
    softmax->Forward = Forward_Softmax;
    int shape[2] = {1,1};
    softmax->input = Init_Matrix(shape, 2, "null");
    softmax->Free = Free_Softmax;
    softmax->output_size = 0;
    Layer* layer = Init_Layer();
    layer->inheritance.softmax = softmax;
    layer->layer = "softmax";
    return layer; 
}
