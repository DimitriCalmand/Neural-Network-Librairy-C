#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "Math_Utils.h"
#include "../Matrix.h"
#include "../utils/Copy.h"
#include <string.h>
#include <time.h>

Matrix* __Exponential_Matrix(Matrix this)
{
    /*
    Return a new matrix with the exponential of each element of the matrix
    @param
    THIS: The matrix
    @return
    The new matrix
    */
    Matrix* res = Init_Matrix(this.shape, this.size_shape, "null");
    #if USE_SIMD == 2
    {
        vdExp(this.size_array, this.array, res->array);
    }
    #else 
    {
        for (int i = 0; i!=this.size_array; i++)
        {
            res->array[i] = exp(this.array[i]);
        }
    }
    #endif
    return res;
}
void Exponential_Matrix(Matrix* this)
{
    /*
    Apply the exponential to each element of the matrix
    @param
    THIS: The matrix
    */
    #if USE_SIMD == 2
    {
        vdExp(this->size_array, this->array, this->array);
    }
    #else 
    {
        for (int i = 0; i!=this->size_array; i++)
        {
            this->array[i] = exp(this->array[i]);
        }
    }
    #endif
}
