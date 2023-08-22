#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "Math_Utils.h"
#include "../Matrix.h"
#include "../utils/Copy.h"
#include <string.h>
#include <time.h>

Matrix* __Abs_Matrix(Matrix this)
{
    /*
    Return a new matrix with the absolute value of each element of the matrix
    @param
    THIS: The matrix
    @return
    The new matrix
    */
    Matrix* res = Init_Matrix(this.shape, this.size_shape, "null");
    for (int i = 0; i!=this.size_array; i++)
    {
        res->array[i] = fabs(this.array[i]);
    }
    return res;
}
void Abs_Matrix(Matrix* this)
{
    /*
    Apply the absolute value to each element of the matrix
    @param
    THIS: The matrix
    */
    for (int i = 0; i!=this->size_array; i++)
    {
        this->array[i] = fabs(this->array[i]);
    }
}