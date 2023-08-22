#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../Matrix.h"
#include <string.h>
#include <time.h>

void Copy_Matrix(Matrix* m1, Matrix* m2)
{
    /*
    Copy the matrix
    @param
    M1: The matrix
    M2: The matrix to copy
    */
    m1->array = realloc(m1->array, m2->size_array * sizeof(VAR));
    m1->size_array = m2->size_array;
    m1->size_shape = m2->size_shape;
    m1->shape = realloc(m1->shape, m1->size_shape * sizeof(int));
    for (int i = 0; i!=m1->size_shape; i++)
    {
        m1->shape[i] = m2->shape[i];
    }

    memcpy(m1->array, m2->array, m2->size_array * sizeof(VAR));

}

Matrix* __Copy_Matrix(Matrix this)
{
    /*
    Return a new matrix with the copy of the matrix
    @param
    THIS: The matrix
    @return
    The new matrix
    */
   //Copy using _m256 simd
    Matrix* res = Init_Matrix(this.shape, this.size_shape, "null");
    memcpy(res->array, this.array, this.size_array * sizeof(VAR));
    return res;

}
