#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../Matrix.h"
#include <string.h>
#include <time.h>


Matrix* __Reshape_Matrix(Matrix this, int* new_shape, int size_new_shape){
    /*
    Return a new matrix with the new shape
    @param
    THIS: The matrix
    NEW_SHAPE: The new shape
    SIZE_NEW_SHAPE: The size of the new shape
    @return
    The new matrix
    */
    int size_array = 1;
    for (int i = 0; i!=size_new_shape; i++)
    {
        size_array *= new_shape[i];
    }
    if (size_array!=this.size_array)
    {
        errx(EXIT_FAILURE, "the matrix doesn't have the correct dimensions for reshape find shape 1: %d, shape 2: %d", this.size_array, size_array);
    }
    Matrix* res = Init_Matrix(new_shape, size_new_shape, "null");
    for (int i = 0; i!=res->size_array; i++)
    {
        res->array  [i] = this.array[i];
    }
    return res;
}
void Reshape_Matrix(Matrix* this, int* new_shape, int size_new_shape)
{
    /*
    Reshape the matrix
    @param
    THIS: The matrix
    NEW_SHAPE: The new shape
    SIZE_NEW_SHAPE: The size of the new shape
    */
    int size_array = 1;
    for (int i = 0; i!=size_new_shape; i++)
    {
        size_array *= new_shape[i];
    }
    if (size_array!=this->size_array)
    {
        errx(EXIT_FAILURE, "the matrix doesn't have the correct dimensions for reshape find shape 1: %d, shape 2: %d", this->size_array, size_array);
    }
    free(this->shape);
    int* shape = malloc(size_new_shape*sizeof(int));
    for (int i = 0; i!=size_new_shape; i++)
    {
        shape[i] = new_shape[i];
    }
    this->size_shape = size_new_shape;
    this->shape = shape;
}
