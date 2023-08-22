#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "Math_Utils.h"
#include "../Matrix.h"
#include "../utils/Copy.h"
#include <string.h>
#include <time.h>

Matrix* __Logarithm_Matrix(Matrix this)
{
    /*
    Return a new matrix with the logarithm of each element of the matrix
    @param
    THIS: The matrix
    @return
    The new matrix
    */
    Matrix* res = Init_Matrix(this.shape, this.size_shape, "null");
    for (int i = 0; i!=this.size_array; i++)
    {
        res->array[i] = log(this.array[i]);
    }
    return res;
}
void Logarithm_Matrix(Matrix* this)
{
    /*
    Apply the logarithm to each element of the matrix
    @param
    THIS: The matrix
    */
    for (int i = 0; i!=this->size_array; i++)
    {
        this->array[i] = log(this->array[i]);
    }
}